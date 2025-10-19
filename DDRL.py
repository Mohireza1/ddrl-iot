from __future__ import annotations

import math
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
from tqdm import tqdm

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import torch.nn as nn
import torch.optim as optim

DEFAULT_CONFIG = {
    "num_rrh": 2,
    "num_devices": 15,
    "num_subch": 4,
    "bandwidth": 40e6,
    "noise": 1e-13,
    "V": 2,
    "f_m": [2.0e9] * 15,
    "L_l": [1.0e-9] * 15,
    "xi": 1.0e-27,
    "psi_lo": [1.0e8] * 15,
    "psi_off": [5.0e7] * 15,
    "d_m": [1.0e6] * 15,
    "p_max": 2.0,
    "r_min": 1.0,
    "Ik_value": 0.5,  # adjusted for power-domain interference
    "omega": 1.0,
    "mu": 1.0,
    "episode_len": 200,
    "num_power_levels": 11,
    "p_c": 0.1,
    "p_s": 0.5,
    "epsilon": 0.5,
    "P_T": 0.5,
    "invalid_penalty": -1.0,
    "E_max": 10.0,
    "E_init_frac": 0.5,
    "Phi_max": [0.1] * 15,
    "seed": 69,
}


def sinr(p, g_pow, interference_pow, sigma2):
    return (abs(p) * g_pow) / (interference_pow + sigma2)


def rate(W, phi):
    return W * np.log2(1.0 + max(phi, 0.0))


def queue_update(q, b, A):
    return max(q - b, 0.0) + A


def rrh_total_power_model(p_sum_rrh, p_c, p_s, eps, n_active_links=0):
    return float(p_c + n_active_links * p_s + eps * p_sum_rrh)


def virtual_power_update(H, P_tilde, P_T):
    return max(H - P_tilde + P_T, 0.0)


def E_local(xi, f_m, L_l):
    return float(xi * (f_m**2) * L_l)


def T_local(f_m, psi_lo_m):
    return float(psi_lo_m / max(f_m, 1e-12))


def E_offload(p_m, f_m, psi_off_m):
    p_tx = abs(p_m)
    return float((p_tx / max(f_m, 1e-12)) * psi_off_m)


def T_offload(p_m, f_m, psi_off_m, d_m, r_m):
    p_tx = abs(p_m)
    t_comp = (p_tx / max(f_m, 1e-12)) * psi_off_m
    t_tx = d_m / max(r_m, 1e-12)
    return float(t_comp + t_tx)


def eta_EE(omega, sum_rate, mu, sum_Po):
    denom = max(mu * sum_Po, 1e-20)
    return (omega * sum_rate) / denom


def _ensure_vector(x, length, name="vector"):
    arr = np.array(x, dtype=float)
    if arr.ndim == 0:
        return np.full(length, float(arr))
    if arr.size == 1:
        return np.full(length, float(arr.ravel()[0]))
    arr = arr.ravel()
    if arr.size != length:
        raise ValueError(f"{name} must be scalar or length {length}, got {arr.size}")
    return arr


def lyapunov_L(Q, H):
    return float(0.5 * (np.sum(Q**2) + np.sum(H**2)))


class MECEnvDDRL(gym.Env):

    def __init__(self, config=None):
        cfg = {**DEFAULT_CONFIG, **(config or {})}

        self.L = int(cfg.get("num_rrh"))
        self.M = int(cfg.get("num_devices"))
        self.K = int(cfg.get("num_subch"))
        self.W_total = float(cfg.get("bandwidth"))
        self.W_sub = self.W_total / self.K
        self.sigma2 = float(cfg.get("noise"))
        self.V = float(cfg.get("V"))

        self.f_m = _ensure_vector(cfg.get("f_m"), self.M, name="f_m")
        self.L_l = _ensure_vector(cfg.get("L_l"), self.M, name="L_l")
        self.xi = float(cfg.get("xi"))
        self.psi_lo = _ensure_vector(cfg.get("psi_lo"), self.M, name="psi_lo")
        self.psi_off = _ensure_vector(cfg.get("psi_off"), self.M, name="psi_off")
        self.d_m = _ensure_vector(cfg.get("d_m"), self.M, name="d_m")

        self._r_m_last = np.zeros(self.M, dtype=float)
        self._p_m_last = np.zeros(self.M, dtype=float)

        self.p_max = float(cfg.get("p_max"))
        self.r_min = float(cfg.get("r_min"))
        Ik_default = np.full(self.K, cfg.get("Ik_value"), dtype=float)
        self.Ik = _ensure_vector(cfg.get("Ik", Ik_default), self.K, name="Ik")

        self.omega = _ensure_vector(cfg.get("omega", 1.0), self.M, name="omega")
        self.mu = _ensure_vector(cfg.get("mu", 1.0), self.L, name="mu")

        self.episode_len = int(cfg.get("episode_len"))
        # channel coherence control
        self.coherence_N = int(cfg.get("coherence_N", 10))
        self._coh_ctr = 0
        self.num_power_levels = int(cfg.get("num_power_levels"))
        p_nz = np.linspace(
            -self.p_max, self.p_max, self.num_power_levels - 1, dtype=float
        )
        # ensure there is a 0 level
        self.power_levels = np.zeros(self.num_power_levels, dtype=np.float32)
        self.power_levels[1:] = p_nz

        self.p_c = float(cfg.get("p_c"))
        self.p_s = float(cfg.get("p_s"))
        self.eps = float(cfg.get("epsilon"))
        self.P_T = float(cfg.get("P_T"))

        self.Q = np.zeros((self.L, self.M), dtype=float)
        self.H = np.zeros((self.L, self.M), dtype=float)
        self.h = np.zeros((self.L, self.M), dtype=float)
        self.Qloc = np.zeros(self.M, dtype=float)

        # Battery energy per device (paper's e_m). We track a simple harvest-storage model.
        self.E_max = float(cfg.get("E_max", 10.0))
        self.E_init_frac = float(cfg.get("E_init_frac", 0.5))
        # Upper bound on harvest per slot for each device
        self.Phi_max = np.array(cfg.get("Phi_max", [0.1] * self.M), dtype=float)
        self.E_m = np.full(self.M, self.E_max * self.E_init_frac, dtype=float)

        self._phi_rrh = np.zeros(self.L, dtype=float)
        self._e_m_obs = np.zeros(self.M, dtype=float)

        obs_dim = self.L + self.M
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        self._choices_per_lk = 1 + self.M * (self.num_power_levels - 1)
        self.action_space = spaces.MultiDiscrete(
            [self._choices_per_lk] * (self.L * self.K)
        )

        self.invalid_penalty = float(cfg.get("invalid_penalty"))

        self._rng = np.random.default_rng(cfg.get("seed"))
        self._t = 0
        self._episode_steps = 0
        self._ep_reward_sum = 0.0
        self._ep_reward_min = np.inf
        self._ep_reward_max = -np.inf

        # Redraw channels only every coherence_N steps
        self._draw_channels()
        self._coh_ctr = 0

    def _draw_channels(self):
        self.h[:, :] = self._rng.rayleigh(scale=1.0, size=(self.L, self.M))

    def _build_obs(self):
        phi_norm = (np.tanh(self._phi_rrh) + 1.0) / 2.0
        e_norm = (self._e_m_obs / max(self.E_max, 1e-12)).clip(0.0, 1.0)
        return np.concatenate([phi_norm.ravel(), e_norm.ravel()]).astype(np.float32)

    def _decode_joint_action(self, a_vec):
        m_sel = -np.ones((self.L, self.K), dtype=int)
        p_sel = np.zeros((self.L, self.K), dtype=float)
        for l in range(self.L):
            for k in range(self.K):
                idx = int(a_vec[l * self.K + k])
                if idx == 0:  # not active
                    continue
                idx -= 1
                m = idx // (self.num_power_levels - 1)
                p_idx_nz = (idx % (self.num_power_levels - 1)) + 1
                m_sel[l, k] = m
                p_sel[l, k] = float(self.power_levels[p_idx_nz])
        return m_sel, p_sel

    def _interference_matrix(self, m_sel, p_sel):
        I = np.zeros((self.L, self.K), dtype=float)
        for k in range(self.K):
            for l in range(self.L):
                m = m_sel[l, k]
                if m < 0:
                    continue
                itf = 0.0
                for lp in range(self.L):
                    if lp == l:
                        continue
                    if m_sel[lp, k] >= 0:
                        p_lp = p_sel[lp, k]
                        # interference from RRH lp to receiver m on subchannel k (power domain)
                        itf += abs(p_lp) * (abs(self.h[lp, m]) ** 2)
                I[l, k] = itf
        return I

    def _interference_matrix_linear(self, m_sel, p_sel):
        # (amplitude-domain)
        I = np.zeros((self.L, self.K), dtype=float)
        for k in range(self.K):
            for l in range(self.L):
                m = m_sel[l, k]
                if m < 0:
                    continue
                itf = 0.0
                for lp in range(self.L):
                    if lp == l:
                        continue
                    if m_sel[lp, k] >= 0:
                        p_lp = p_sel[lp, k]
                        itf += abs(p_lp) * abs(self.h[lp, m])  # amplitude
                I[l, k] = itf
        return I

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._t = 0
        self._episode_steps = 0
        self.Q.fill(0.0)
        self.H.fill(0.0)
        self.Qloc.fill(0.0)
        # Reset observation state vectors
        self._phi_rrh[:] = 0.0
        self._e_m_obs[:] = 0.0
        # Reset batteries
        self.E_m[:] = self.E_max * self.E_init_frac
        self._e_m_obs[:] = self.E_m.copy()
        # Redraw channels only every coherence_N steps
        self._coh_ctr += 1
        if self._coh_ctr >= self.coherence_N:
            self._draw_channels()
            self._coh_ctr = 0
        self._ep_reward_sum = 0.0
        self._ep_reward_min = np.inf
        self._ep_reward_max = -np.inf
        obs = self._build_obs()
        info = {
            "lyapunov": lyapunov_L(self.Q, self.H),
            "t": self._t,
            "episode_steps": self._episode_steps,
            "phi_rrh": self._phi_rrh.copy(),
            "e_m": self._e_m_obs.copy(),
        }
        return obs, info

    def step(self, action):
        m_sel, p_sel = self._decode_joint_action(action)

        for m in range(self.M):
            used_pairs = [
                (l, k) for l in range(self.L) for k in range(self.K) if m_sel[l, k] == m
            ]
            if len(used_pairs) > 1:
                obs = self._build_obs()
                return (
                    obs,
                    self.invalid_penalty,
                    False,
                    False,
                    {
                        "ok": False,
                        "constraint": "C4_single_association",
                        "device": int(m),
                        "pairs": used_pairs,
                    },
                )

        for l in range(self.L):
            p_sum = rrh_total_power(p_sel[l, :])
            if p_sum > self.p_max + 1e-12:
                obs = self._build_obs()
                return (
                    obs,
                    self.invalid_penalty,
                    False,
                    False,
                    {
                        "ok": False,
                        "constraint": "C1_power_cap",
                        "l": int(l),
                        "p_sum": float(p_sum),
                        "p_max": float(self.p_max),
                    },
                )

        I_pow = self._interference_matrix(m_sel, p_sel)  # power-domain (|h|^2) for SINR
        I_chk = self._interference_matrix_linear(
            m_sel, p_sel
        )  # amplitude-domain (|h|) for constraint (5)

        for l in range(self.L):
            for k in range(self.K):
                if I_chk[l, k] > float(self.Ik[k]) + 1e-12:
                    obs = self._build_obs()
                    return (
                        obs,
                        self.invalid_penalty,
                        False,
                        False,
                        {
                            "ok": False,
                            "constraint": "C3_interference",
                            "l": int(l),
                            "k": int(k),
                            "lhs": float(I_chk[l, k]),
                            "rhs": float(self.Ik[k]),
                        },
                    )

        phi = np.zeros((self.L, self.K), dtype=float)
        r = np.zeros((self.L, self.K), dtype=float)
        for l in range(self.L):
            for k in range(self.K):
                m = m_sel[l, k]
                if m < 0:
                    continue
                p = p_sel[l, k]
                g_pow = abs(self.h[l, m]) ** 2
                phi[l, k] = sinr(p, g_pow, I_pow[l, k], self.sigma2)
                r[l, k] = rate(self.W_sub, phi[l, k])

        r_m_total = np.zeros(self.M, dtype=float)
        for l in range(self.L):
            for k in range(self.K):
                m = m_sel[l, k]
                if m >= 0:
                    r_m_total[m] += r[l, k]
        for m in range(self.M):
            if r_m_total[m] > 0.0 and r_m_total[m] < self.r_min - 1e-12:
                obs = self._build_obs()
                return (
                    obs,
                    self.invalid_penalty,
                    False,
                    False,
                    {
                        "ok": False,
                        "constraint": "C2_rate_min_agg",
                        "device": int(m),
                        "rate_total": float(r_m_total[m]),
                        "r_min": float(self.r_min),
                    },
                )

        rrh_p_sums = np.array(
            [rrh_total_power(p_sel[l, :]) for l in range(self.L)], dtype=float
        )

        n_active = np.array(
            [int(np.count_nonzero(m_sel[l, :] >= 0)) for l in range(self.L)], dtype=int
        )
        rrh_Po = np.array(
            [
                rrh_total_power_model(
                    rrh_p_sums[l],
                    self.p_c,
                    self.p_s,
                    self.eps,
                    n_active_links=int(n_active[l]),
                )
                for l in range(self.L)
            ],
            dtype=float,
        )

        sum_rates = float(np.sum(r))
        sum_Po = float(np.sum(rrh_Po))

        r_m = np.zeros(self.M, dtype=float)
        p_m = np.zeros(self.M, dtype=float)
        for l in range(self.L):
            for k in range(self.K):
                m = m_sel[l, k]
                if m >= 0:
                    r_m[m] += r[l, k]
                    p_m[m] += abs(p_sel[l, k])

        L_prev = lyapunov_L(self.Q, self.H)

        # Which devices are actually offloaded this step
        scheduled_any = np.zeros(self.M, dtype=bool)
        for l in range(self.L):
            for k in range(self.K):
                m = m_sel[l, k]
                if m >= 0 and p_sel[l, k] > 1e-12:
                    scheduled_any[m] = True

        # per-device arrivals
        A_m = self._rng.uniform(0.0, 0.1, size=self.M)

        A = np.zeros((self.L, self.M), dtype=float)
        for m in range(self.M):
            if scheduled_any[m]:
                # send to the first RRH that selected m
                for l in range(self.L):
                    if m in m_sel[l, :]:
                        A[l, m] = A_m[m]
                        break
        A_local_in = A_m * (~scheduled_any)

        # MEC service b[l,m]
        b = np.zeros((self.L, self.M), dtype=float)
        for l in range(self.L):
            for k in range(self.K):
                m = m_sel[l, k]
                if m >= 0:
                    b[l, m] += r[l, k]

        served_mec_mat = np.minimum(self.Q, b)
        mec_served_bits = float(np.sum(served_mec_mat))

        for l in range(self.L):
            for m in range(self.M):
                self.Q[l, m] = queue_update(self.Q[l, m], b[l, m], A[l, m])

        # capacity
        C_loc = self.f_m / np.maximum(self.psi_lo, 1e-12)  # shape (M,)
        # served locally
        served_local = np.minimum(self.Qloc, C_loc)
        local_served_bits = float(np.sum(served_local))
        # local backlog update: drain then add arrivals not offloaded
        self.Qloc = np.maximum(self.Qloc - C_loc, 0.0) + A_local_in

        # Devices with any MEC backlog (for a rough utilization proxy)
        pending_devices = np.any(self.Q > 1e-12, axis=0)
        pending_mec_pct = 100.0 * np.mean(pending_devices)

        # Update paper-state proxies for next observation per Eq. (25)
        # phi_l = average SINR across active subchannels at RRH l (0 if none)
        for l in range(self.L):
            active_mask = m_sel[l, :] >= 0
            if np.any(active_mask):
                self._phi_rrh[l] = float(np.mean(phi[l, active_mask]))
            else:
                self._phi_rrh[l] = 0.0
        # e_m = battery energy level for device m
        self._e_m_obs[:] = self.E_m.copy()

        scheduled = (r_m > 1e-12) | (p_m > 1e-12)

        E_lo_m = np.array(
            [E_local(self.xi, self.f_m[m], self.L_l[m]) for m in range(self.M)],
            dtype=float,
        )
        T_lo_m = np.array(
            [T_local(self.f_m[m], self.psi_lo[m]) for m in range(self.M)], dtype=float
        )
        E_off_m = np.array(
            [E_offload(p_m[m], self.f_m[m], self.psi_off[m]) for m in range(self.M)],
            dtype=float,
        )

        T_off_m = np.zeros(self.M, dtype=float)
        for m in range(self.M):
            if scheduled[m]:
                T_off_m[m] = T_offload(
                    p_m[m], self.f_m[m], self.psi_off[m], self.d_m[m], r_m[m]
                )
            else:
                T_off_m[m] = 0.0

        E_T_m = E_lo_m + E_off_m

        num = float(np.dot(self.omega, r_m))
        den = float(np.dot(self.mu, rrh_Po))
        eta = num / max(den, 1e-20)

        harvest = self._rng.uniform(0.0, 1.0, size=self.M) * self.Phi_max
        self.E_m = np.clip(self.E_m - E_T_m + harvest, 0.0, self.E_max)

        P_lm = np.zeros((self.L, self.M), dtype=float)
        for l in range(self.L):
            for k in range(self.K):
                m = m_sel[l, k]
                if m >= 0:
                    P_lm[l, m] += max(p_sel[l, k], 0.0)
        # Convert to \tilde P via epsilon and update H
        for l in range(self.L):
            for m in range(self.M):
                if P_lm[l, m] > 0.0 or self.P_T != 0.0:
                    P_tilde = P_lm[l, m] / max(self.eps, 1e-12)
                    self.H[l, m] = virtual_power_update(self.H[l, m], P_tilde, self.P_T)
        Lphi = lyapunov_L(self.Q, self.H)
        delta_L = Lphi - L_prev
        reward_rl = (self.V * eta - delta_L) / 1e8
        self._r_m_last[:] = r_m
        self._p_m_last[:] = p_m
        self._ep_reward_sum += reward_rl
        self._ep_reward_min = min(self._ep_reward_min, reward_rl)
        self._ep_reward_max = max(self._ep_reward_max, reward_rl)

        self._episode_steps += 1
        self._t += 1
        self._coh_ctr += 1
        if self._coh_ctr >= self.coherence_N:
            self._draw_channels()
            self._coh_ctr = 0

        obs = self._build_obs()
        terminated = False
        truncated = self._episode_steps >= self.episode_len
        info = {
            "ok": True,
            "eta_EE": float(eta),
            "sum_rates": sum_rates,
            "sum_Po": sum_Po,
            "lyapunov": float(Lphi),
            "t": self._t,
            "delta_L": float(delta_L),
            "dpp_obj": float(delta_L - self.V * eta),
            "r_m": r_m.tolist(),
            "p_m": p_m.tolist(),
            "mec_tasks": int(np.count_nonzero(scheduled)),  # devices scheduled (proxy)
            "T_off_avg": (
                float(np.mean(T_off_m[scheduled])) if np.any(scheduled) else 0.0
            ),
            "E_total_sum": float(np.sum(E_T_m)),
            "pending_mec_pct": float(pending_mec_pct),
            "offloaded_pct": 100.0 * (np.count_nonzero(scheduled) / self.M),
            "mec_served_bits": float(mec_served_bits),
            "local_served_bits": float(local_served_bits),
            "mec_task_pct": 100.0
            * float(mec_served_bits)
            / max(float(mec_served_bits + local_served_bits), 1e-12),
        }
        return obs, reward_rl, terminated, truncated, info


def rrh_total_power(p_vec_for_rrh):
    return float(np.sum(np.abs(p_vec_for_rrh)))


def safe_random_action(env: MECEnvDDRL):
    a = np.zeros(env.L * env.K, dtype=int)
    l = np.random.randint(env.L)
    k = np.random.randint(env.K)
    m = np.random.randint(env.M)
    target_p = min(env.p_max * 0.25, env.power_levels[-1])
    p_idx = int(np.argmin(np.abs(env.power_levels - target_p)))
    p_idx = max(1, p_idx)
    idx_nonzero = m * (env.num_power_levels - 1) + (p_idx - 1)
    encoded = 1 + idx_nonzero
    a[l * env.K + k] = encoded
    return a


@dataclass
class EpisodeLog:
    episode: int
    steps: int = 0
    rewards: List[float] = field(default_factory=list)
    per_step: List[Dict[str, Any]] = field(default_factory=list)

    def add_step(self, r: float, info: Dict[str, Any]):
        r_m = info.get("r_m", [])
        p_m = info.get("p_m", [])
        try:
            import numpy as _np

            _r = _np.array(r_m, dtype=float)
            _p = _np.array(p_m, dtype=float)
            mec_tasks = int(_np.count_nonzero((_r > 1e-12) | (_p > 1e-12)))
        except Exception:
            mec_tasks = 0
        self.rewards.append(float(r))
        self.per_step.append(
            {
                "t": int(info.get("t", self.steps)),
                "eta_EE": float(info.get("eta_EE", 0.0)),
                "delta_L": float(info.get("delta_L", 0.0)),
                "dpp_obj": float(info.get("dpp_obj", 0.0)),
                "sum_rates": float(info.get("sum_rates", 0.0)),
                "sum_Po": float(info.get("sum_Po", 0.0)),
                "lyapunov": float(info.get("lyapunov", 0.0)),
                "E_total_sum": float(info.get("E_total_sum", 0.0)),
                "T_off_avg": float(info.get("T_off_avg", 0.0)),
                "loss": float(info.get("loss", float("nan"))),
                "mec_tasks": mec_tasks,
                "pending_mec_pct": float(info.get("pending_mec_pct", 0.0)),
                "offloaded_pct": float(info.get("offloaded_pct", 0.0)),
                "mec_task_pct": float(info.get("mec_task_pct", float("nan"))),
                "mec_bits": float(info.get("mec_bits", 0.0)),
                "local_bits": float(info.get("local_bits", 0.0)),
                "arr_bits": float(info.get("arr_bits", 0.0)),
                "mec_served_bits": float(info.get("mec_served_bits", 0.0)),
                "local_served_bits": float(info.get("local_served_bits", 0.0)),
                "mec_task_pct": float(info.get("mec_task_pct", float("nan"))),
            }
        )
        self.steps += 1

    def episode_reward(self) -> float:
        return float(sum(self.rewards))


class EnvLogger:
    def __init__(self):
        self.episodes: List[EpisodeLog] = []
        self._current: Optional[EpisodeLog] = None

    def start_episode(self, episode_idx: int):
        if self._current is not None:
            raise RuntimeError("Episode already started; call end_episode() first.")
        self._current = EpisodeLog(episode=episode_idx)

    def log_step(self, reward: float, info: Dict[str, Any]):
        if self._current is None:
            raise RuntimeError("No episode started; call start_episode().")
        self._current.add_step(reward, info)

    def end_episode(self):
        if self._current is None:
            raise RuntimeError("No active episode to end.")
        self.episodes.append(self._current)
        self._current = None

    def save_episode_summaries(self, path: str):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "steps",
                    "reward_sum",
                    "reward_mean",
                    "reward_min",
                    "reward_max",
                ]
            )
            for ep in self.episodes:
                rs = ep.rewards
                writer.writerow(
                    [
                        ep.episode,
                        ep.steps,
                        sum(rs),
                        (sum(rs) / len(rs)) if rs else 0.0,
                        min(rs) if rs else 0.0,
                        max(rs) if rs else 0.0,
                    ]
                )

    def save_per_step(self, path: str):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "episode",
                    "t",
                    "reward",
                    "eta_EE",
                    "delta_L",
                    "dpp_obj",
                    "sum_rates",
                    "sum_Po",
                    "lyapunov",
                    "E_total_sum",
                    "T_off_avg",
                    "loss",
                    "mec_tasks",
                    "pending_mec_pct",
                    "offloaded_pct",
                    "mec_bits",
                    "arr_bits",
                    "mec_task_pct",
                    "pending_mec_pct",
                    "offloaded_pct",
                    "mec_served_bits",
                    "local_served_bits",
                    "mec_task_pct",
                ]
            )
            for ep in self.episodes:
                for r, row in zip(ep.rewards, ep.per_step):
                    writer.writerow(
                        [
                            ep.episode,
                            row["t"],
                            r,
                            row["eta_EE"],
                            row["delta_L"],
                            row["dpp_obj"],
                            row["sum_rates"],
                            row["sum_Po"],
                            row["lyapunov"],
                            row["E_total_sum"],
                            row["T_off_avg"],
                            row["loss"],
                            row["mec_tasks"],
                            row["pending_mec_pct"],
                            row["offloaded_pct"],
                            row.get("mec_bits", 0.0),
                            row.get("arr_bits", 0.0),
                            row.get("mec_task_pct", 0.0),
                            row.get("pending_mec_pct", 0.0),
                            row.get("offloaded_pct", 0.0),
                            row.get("mec_served_bits", 0.0),
                            row.get("local_served_bits", 0.0),
                            row.get("mec_task_pct", 0.0),
                        ]
                    )

    def save_per_step_normalized(
        self, path: str, scale_eta=1e6, scale_rates=1e6, scale_dpp=1e8
    ):
        import csv

        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "episode",
                    "t",
                    "reward",  # reward already scaled in env
                    "eta_EE_Mb_per_J",  # eta_EE / scale_eta
                    "delta_L",
                    "dpp_obj_scaled",  # dpp_obj / scale_dpp
                    "sum_rates_Mbps",  # sum_rates / scale_rates
                    "sum_Po_W",
                    "lyapunov",
                    "E_total_sum",
                    "T_off_avg",
                    "loss",
                    "mec_tasks",
                    "offloaded_pct",  # <— added
                ]
            )
            for ep in self.episodes:
                for r, row in zip(ep.rewards, ep.per_step):
                    w.writerow(
                        [
                            ep.episode,
                            row["t"],
                            r,
                            row["eta_EE"] / scale_eta,
                            row["delta_L"],
                            row["dpp_obj"] / scale_dpp,
                            row["sum_rates"] / scale_rates,
                            row["sum_Po"],
                            row["lyapunov"],
                            row["E_total_sum"],
                            row["T_off_avg"],
                            row["loss"],
                            row["mec_tasks"],
                            row["offloaded_pct"],
                        ]
                    )


class FactorizedDuelingQ(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        num_pos: int,
        choices_per_pos: int,
        hid1: int = 120,
        hid2: int = 80,
    ):
        super().__init__()
        self.num_pos = num_pos
        self.choices = choices_per_pos
        self.trunk = nn.Sequential(
            nn.LayerNorm(obs_dim),
            nn.Linear(obs_dim, hid1),
            nn.ReLU(inplace=True),
            nn.Linear(hid1, hid2),
            nn.ReLU(inplace=True),
        )
        self.V = nn.Linear(hid2, 1)
        self.A = nn.Linear(hid2, num_pos * choices_per_pos)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(x)
        V = self.V(z)
        A = self.A(z).view(-1, self.num_pos, self.choices)
        A_mean = A.mean(dim=-1, keepdim=True)
        Q = V.unsqueeze(-1) + (A - A_mean)
        return Q, V


class Replay:
    def __init__(self, capacity=400):
        self.capacity = capacity
        self.buf = []
        self.idx = 0

    def push(self, s, a_vec, r, s2, done):
        data = (s.copy(), a_vec.copy(), float(r), s2.copy(), bool(done))
        if len(self.buf) < self.capacity:
            self.buf.append(data)
        else:
            self.buf[self.idx] = data
        self.idx = (self.idx + 1) % self.capacity

    def sample(self, batch_size: int):
        idxs = np.random.choice(len(self.buf), size=batch_size, replace=False)
        batch = [self.buf[i] for i in idxs]
        s, a, r, s2, d = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.array(s2, dtype=np.float32),
            np.array(d, dtype=np.uint8),
        )

    def __len__(self):
        return len(self.buf)


class DuelingDoubleDQNAgent:
    def __init__(
        self,
        obs_dim,
        num_pos,
        choices_per_pos,
        lr=1e-2,
        gamma=0.9,
        device="cpu",
        hid1=120,
        hid2=80,
        target_tau=0.005,
    ):
        self.device = torch.device(device)
        self.gamma = gamma
        self.num_pos = num_pos
        self.choices = choices_per_pos
        self.net = FactorizedDuelingQ(obs_dim, num_pos, choices_per_pos, hid1, hid2).to(
            self.device
        )
        self.tgt = FactorizedDuelingQ(obs_dim, num_pos, choices_per_pos, hid1, hid2).to(
            self.device
        )
        self.tgt.load_state_dict(self.net.state_dict())
        self.opt = optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-3)
        self.replay = Replay(capacity=400)
        self.tau = target_tau
        self.loss_fn = nn.MSELoss(reduction="mean")
        self.eps_start = 1.0
        self.eps_final = 0.05
        self.eps_decay = 2000
        self._steps = 0

    def epsilon(self):
        return self.eps_final + (self.eps_start - self.eps_final) * math.exp(
            -self._steps / self.eps_decay
        )

    def push(self, s, a_vec, r, s2, done):
        self.replay.push(s, a_vec, r, s2, done)

    def train_step(self, batch_size=10):
        if len(self.replay) < batch_size:
            return None
        s, a_vec, r, s2, d = self.replay.sample(batch_size)
        s = torch.tensor(s, dtype=torch.float32, device=self.device)
        a = torch.tensor(a_vec, dtype=torch.int64, device=self.device)
        r = torch.tensor(r, dtype=torch.float32, device=self.device)
        s2 = torch.tensor(s2, dtype=torch.float32, device=self.device)
        d = torch.tensor(d, dtype=torch.float32, device=self.device)

        Q_all, _ = self.net(s)
        Qa = Q_all.gather(dim=2, index=a.unsqueeze(-1)).squeeze(-1)
        Qa_sum = Qa.sum(dim=1)

        with torch.no_grad():
            Q_next_all, _ = self.net(s2)
            next_a = torch.argmax(Q_next_all, dim=-1)
            Q_next_tgt_all, _ = self.tgt(s2)
            Q_next = Q_next_tgt_all.gather(2, next_a.unsqueeze(-1)).squeeze(-1)
            Q_next_sum = Q_next.sum(dim=1)
            y = r + (1.0 - d) * self.gamma * Q_next_sum

        loss = self.loss_fn(Qa_sum, y)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=5.0)
        self.opt.step()

        with torch.no_grad():
            for p, tp in zip(self.net.parameters(), self.tgt.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * p.data)
        return float(loss.item())

    def _candidate_list(self, Q_1_P_C: torch.Tensor) -> np.ndarray:
        Q = Q_1_P_C.squeeze(0)
        A = Q[:, 1:]
        flat_idx = torch.argsort(A.reshape(-1), descending=True)
        P, Cminus1 = A.shape
        pos = (flat_idx // Cminus1).cpu().numpy()
        ch_rel = (flat_idx % Cminus1).cpu().numpy()
        choice = ch_rel + 1
        return np.stack([pos, choice], axis=1)

    def act(self, obs: np.ndarray, env) -> np.ndarray:
        self._steps += 1
        eps = self.epsilon()
        if np.random.rand() < eps:
            P, C = env.L * env.K, env._choices_per_lk
            rand_scores = torch.randn(1, P, C, device=self.device)
            cand = self._candidate_list(rand_scores)
            return self._decode_feasible(env, cand)
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            Q, _ = self.net(x)
            cand = self._candidate_list(Q)
            return self._decode_feasible(env, cand)

    def _decode_feasible(self, env, candidates: np.ndarray) -> np.ndarray:
        L, K, M = env.L, env.K, env.M
        a = np.zeros(L * K, dtype=np.int64)
        m_sel = -np.ones((L, K), dtype=int)
        p_sel = np.zeros((L, K), dtype=float)
        rrh_p_sum = np.zeros(L, dtype=float)
        device_used = np.zeros(M, dtype=bool)

        def Po_of(l, p_sum):
            n_active = int(np.count_nonzero(m_sel[l, :] >= 0))
            return rrh_total_power_model(
                p_sum, env.p_c, env.p_s, env.eps, n_active_links=n_active
            )

        rrh_Po = np.array([Po_of(l, rrh_p_sum[l]) for l in range(L)], dtype=float)
        sum_rates = 0.0
        sum_Po = float(np.sum(rrh_Po))
        dLH = 0.0

        def decode_choice(choice_idx: int):
            idx = choice_idx - 1
            m = idx // (env.num_power_levels - 1)
            p_idx_nz = (idx % (env.num_power_levels - 1)) + 1
            return m, float(env.power_levels[p_idx_nz])

        def eta_curr():
            # Use scalar equivalents to avoid broadcasting when omega/mu are vectors
            w_s = float(
                getattr(env.omega, "mean", lambda: env.omega)()
                if hasattr(env.omega, "mean")
                else env.omega
            )
            mu_s = float(
                getattr(env.mu, "mean", lambda: env.mu)()
                if hasattr(env.mu, "mean")
                else env.mu
            )
            return (w_s * sum_rates) / (mu_s * max(sum_Po, 1e-20))

        def S_curr():
            return env.V * eta_curr() - dLH

        for pos, choice_idx in candidates:
            l = int(pos // K)
            k = int(pos % K)
            if choice_idx <= 0:
                continue
            m, p = decode_choice(int(choice_idx))
            if device_used[m]:
                continue
            if rrh_p_sum[l] + abs(p) > env.p_max + 1e-12:
                continue
            Itf_amp = 0.0
            Itf_pow = 0.0
            for lp in range(L):
                if lp == l:
                    continue
                if m_sel[lp, k] >= 0:
                    h_lp_m = abs(env.h[lp, m])  # lp → m
                    Itf_amp += abs(p_sel[lp, k]) * h_lp_m
                    Itf_pow += abs(p_sel[lp, k]) * (h_lp_m**2)
            # Feasibility (C3) is in amplitude domain
            if Itf_amp > float(env.Ik[k]) + 1e-12:
                continue
            h_lm_pow = abs(env.h[l, m]) ** 2
            phi_new = sinr(p, h_lm_pow, Itf_pow, env.sigma2)
            r_new = rate(env.W_sub, phi_new)
            if r_new < env.r_min - 1e-12:
                continue

            ok = True
            for lp in range(L):
                if lp == l:
                    continue
                if m_sel[lp, k] >= 0:
                    m_lp = m_sel[lp, k]
                    p_lp = p_sel[lp, k]
                    Itf_lp_check = 0.0
                    Itf_lp_pow = 0.0
                    for lq in range(L):
                        if lq == lp:
                            continue
                        if m_sel[lq, k] >= 0:
                            h_lp_lq = abs(env.h[lp, m_sel[lq, k]])
                            Itf_lp_check += abs(p_sel[lq, k]) * h_lp_lq
                            Itf_lp_pow += abs(p_sel[lq, k]) * (h_lp_lq**2)
                    h_lp_m = abs(env.h[lp, m])
                    Itf_lp_check += abs(p) * h_lp_m
                    Itf_lp_pow += abs(p) * (h_lp_m**2)
                    if Itf_lp_check > float(env.Ik[k]) + 1e-12:
                        ok = False
                        break
                    h_lp_mlp_pow = abs(env.h[lp, m_lp]) ** 2
                    phi_lp = sinr(p_lp, h_lp_mlp_pow, Itf_lp_pow, env.sigma2)
                    r_lp = rate(env.W_sub, phi_lp)
                    if r_lp < env.r_min - 1e-12:
                        ok = False
                        break
            if not ok:
                continue

            S_before = S_curr()
            sum_rates_new = sum_rates + r_new
            Po_l_before = rrh_Po[l]
            Po_l_after = Po_of(l, rrh_p_sum[l] + abs(p))
            sum_Po_new = sum_Po - Po_l_before + Po_l_after
            # Scalarized eta for the candidate action
            w_s = float(
                getattr(env.omega, "mean", lambda: env.omega)()
                if hasattr(env.omega, "mean")
                else env.omega
            )
            mu_s = float(
                getattr(env.mu, "mean", lambda: env.mu)()
                if hasattr(env.mu, "mean")
                else env.mu
            )
            eta_after = (w_s * sum_rates_new) / (mu_s * max(sum_Po_new, 1e-20))
            V_eta_gain = env.V * (eta_after - eta_curr())

            Hprev = env.H[l, m]
            P_tilde = max(p, 0.0) / max(env.eps, 1e-12)
            Hnext = max(Hprev - P_tilde + env.P_T, 0.0)
            dLH_inc = 0.5 * ((Hnext * Hnext) - (Hprev * Hprev))

            if V_eta_gain - dLH_inc < -1e-12:
                continue

            a[pos] = int(choice_idx)
            m_sel[l, k] = m
            p_sel[l, k] = p
            rrh_p_sum[l] += abs(p)
            rrh_Po[l] = Po_l_after
            sum_rates = sum_rates_new
            sum_Po = sum_Po_new
            dLH += dLH_inc
            device_used[m] = True

        return a


### Training Config

EPISODES = 120
STEPS_PER_E = 200
TRAIN_INTERVAL = 10
WARMUP_STEPS = 400
LEARNING_RATE = 1e-4
BATCH_SIZE = 30
GAMMA = 0.9
SEED = DEFAULT_CONFIG["seed"]


def _safe_random_action(env):
    a = np.zeros(env.L * env.K, dtype=np.int64)
    l = np.random.randint(env.L)
    k = np.random.randint(env.K)
    m = np.random.randint(env.M)
    target_p = min(env.p_max * 0.25, env.power_levels[-1])
    p_idx = int(np.argmin(np.abs(env.power_levels - target_p)))
    p_idx = max(1, p_idx)
    idx_nonzero = m * (env.num_power_levels - 1) + (p_idx - 1)
    encoded = 1 + idx_nonzero
    a[l * env.K + k] = encoded
    return a


def main():
    env = MECEnvDDRL(DEFAULT_CONFIG)
    obs, _ = env.reset(seed=SEED)

    num_pos = env.L * env.K
    choices = env._choices_per_lk
    agent = DuelingDoubleDQNAgent(
        obs_dim=env.observation_space.shape[0],
        num_pos=num_pos,
        choices_per_pos=choices,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        device="cpu",
        hid1=120,
        hid2=80,
    )

    logger = EnvLogger()
    Path("runs").mkdir(exist_ok=True)

    global_step = 0
    for ep in tqdm(range(EPISODES)):
        obs, _ = env.reset()
        logger.start_episode(ep)
        for t in range(STEPS_PER_E):
            a_vec = agent.act(obs, env)
            obs2, r, term, trunc, info2 = env.step(a_vec)
            done = term or trunc
            if info2.get("ok", False):
                agent.push(obs, a_vec, r, obs2, done)
            global_step += 1
            loss_val = None
            if global_step >= WARMUP_STEPS and (global_step % TRAIN_INTERVAL == 0):
                loss_val = agent.train_step(batch_size=BATCH_SIZE)
            if loss_val is not None:
                info2["loss"] = float(loss_val)
            logger.log_step(r, info2)
            obs = obs2
            if done:
                break
        logger.end_episode()

    logger.save_episode_summaries("runs/episode_summaries.csv")
    logger.save_per_step("runs/per_step.csv")
    print("Saved: runs/episode_summaries.csv and runs/per_step.csv")
    logger.save_per_step_normalized(
        "runs/per_step_norm.csv",
        scale_eta=1e6,  # Mb/J
        scale_rates=1e6,  # Mb/s
        scale_dpp=1e8,  # matches current reward denom
    )


if __name__ == "__main__":
    np.random.seed(SEED)
    main()
