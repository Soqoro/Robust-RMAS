from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch


MESSAGE_SITES = {"p2c", "c2s", "s2p"}
STATE_SITES = {"planner_self", "critic_self", "refiner_self", "solver_self"}
TERMINAL_SITES = {"final_c2s"}


@dataclass
class RoleProfileConfig:
    enabled: bool = False
    trace_path: Optional[str] = None
    trace_dtype: str = "float32"
    trace_messages: bool = True
    trace_states: bool = True
    trace_terminal: bool = True
    probe_mode: str = "none"
    probe_target: str = "none"
    probe_site: str = "none"
    probe_round: int = -1
    epsilon: float = 0.0
    seed: int = 42
    direction: str = "random"
    record_actual_delta: bool = True

    def metadata(self) -> Dict[str, Any]:
        return asdict(self)


def canonical_probe_site(site: str) -> str:
    if site == "refiner_self":
        return "critic_self"
    return str(site or "none")


def state_site_to_role(site: str) -> str:
    canonical = canonical_probe_site(site)
    if canonical == "planner_self":
        return "planner"
    if canonical == "critic_self":
        return "critic"
    if canonical == "solver_self":
        return "solver"
    raise ValueError(f"Unsupported role-profile state site: {site!r}")


def resolve_role_profile_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    raise ValueError("--role_profile_trace_dtype must be one of: float32, float16, bfloat16.")


def stable_role_seed(base_seed: int, site: str, round_idx: int, sample_index: int) -> int:
    payload = f"role-profile:{int(base_seed)}:{site}:{int(round_idx)}:{int(sample_index)}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**63)


def per_sample_norm(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 0:
        raise ValueError("Expected a batched tensor for role-profile norm.")
    return torch.linalg.vector_norm(x.reshape(x.size(0), -1), dim=1)


def normalized_random_perturbation(
    x: torch.Tensor,
    base_seed: int,
    site: str,
    round_idx: int,
    batch_start: int,
) -> torch.Tensor:
    if x.dim() < 2:
        raise ValueError(f"Expected a batched tensor with rank >= 2, got {tuple(x.shape)}")
    noises: List[torch.Tensor] = []
    sample_shape = tuple(x.shape[1:])
    for offset in range(x.size(0)):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(stable_role_seed(base_seed, site, round_idx, batch_start + offset))
        noise = torch.randn(sample_shape, generator=generator, dtype=torch.float32, device="cpu")
        noises.append(noise)
    out = torch.stack(noises, dim=0).to(device=x.device)
    norm = per_sample_norm(out).clamp_min(1e-12)
    view_shape = (x.size(0),) + (1,) * (x.dim() - 1)
    return out / norm.view(view_shape)


def should_probe(
    cfg: Optional[RoleProfileConfig],
    target: str,
    site: str,
    round_idx: int,
) -> bool:
    if cfg is None or not cfg.enabled:
        return False
    if cfg.probe_mode != "one_shot":
        return False
    if float(cfg.epsilon) <= 0.0:
        return False
    if str(cfg.probe_target) != str(target):
        return False
    if canonical_probe_site(cfg.probe_site) != canonical_probe_site(site):
        return False
    if int(cfg.probe_round) >= 0 and int(cfg.probe_round) != int(round_idx):
        return False
    return True


def apply_role_profile_probe(
    x: torch.Tensor,
    cfg: Optional[RoleProfileConfig],
    target: str,
    site: str,
    round_idx: int,
    batch_start: int,
    recorder: Optional["RoleProfileTraceRecorder"] = None,
) -> Tuple[torch.Tensor, Optional[Dict[str, Any]]]:
    if not should_probe(cfg, target=target, site=site, round_idx=round_idx):
        return x, None
    if cfg is None:
        return x, None
    direction_name = str(cfg.direction or "random")
    if direction_name != "random":
        raise ValueError(f"Unsupported role-profile probe direction: {direction_name!r}")

    x_float = x.detach().float()
    x_norm = per_sample_norm(x_float)
    if x_float.numel() == 0:
        delta_float = torch.zeros_like(x_float)
    else:
        direction = normalized_random_perturbation(
            x,
            base_seed=int(cfg.seed),
            site=canonical_probe_site(site),
            round_idx=int(round_idx),
            batch_start=int(batch_start),
        )
        view_shape = (x.size(0),) + (1,) * (x.dim() - 1)
        delta_float = float(cfg.epsilon) * x_norm.view(view_shape) * direction
    x_tilde = (x_float + delta_float).to(dtype=x.dtype)
    applied_delta = x_tilde.detach().float() - x_float
    delta_norm = per_sample_norm(applied_delta)
    relative_delta_norm = torch.where(
        x_norm > 0,
        delta_norm / x_norm.clamp_min(1e-12),
        torch.zeros_like(delta_norm),
    )
    if recorder is not None:
        recorder.record_probe_delta(
            site=canonical_probe_site(site),
            round_idx=round_idx,
            batch_start=batch_start,
            delta=applied_delta if bool(cfg.record_actual_delta) else None,
            delta_norm=delta_norm,
            x_norm=x_norm,
        )
    meta = {
        "mode": cfg.probe_mode,
        "target": target,
        "site": canonical_probe_site(site),
        "round_idx": int(round_idx),
        "epsilon": float(cfg.epsilon),
        "direction": direction_name,
        "x_norm_mean": float(x_norm.mean().item()) if x_norm.numel() else 0.0,
        "delta_norm_mean": float(delta_norm.mean().item()) if delta_norm.numel() else 0.0,
        "relative_delta_norm_mean": (
            float(relative_delta_norm.mean().item()) if relative_delta_norm.numel() else 0.0
        ),
        "batch_start": int(batch_start),
    }
    return x_tilde, meta


class RoleProfileTraceRecorder:
    def __init__(
        self,
        path: str,
        metadata: Mapping[str, Any],
        sample_ids: Sequence[str],
        sample_indices: Sequence[int],
        dtype: torch.dtype,
        trace_messages: bool = True,
        trace_states: bool = True,
        trace_terminal: bool = True,
        record_actual_delta: bool = True,
    ) -> None:
        self.path = str(path)
        self.metadata = dict(metadata)
        self.sample_ids = [str(x) for x in sample_ids]
        self.sample_indices = [int(x) for x in sample_indices]
        self.dtype = dtype
        self.trace_messages = bool(trace_messages)
        self.trace_states = bool(trace_states)
        self.trace_terminal = bool(trace_terminal)
        self.record_actual_delta = bool(record_actual_delta)
        self._messages: Dict[str, Dict[int, List[Tuple[int, torch.Tensor]]]] = {}
        self._states: Dict[str, Dict[int, List[Tuple[int, torch.Tensor]]]] = {}
        self._terminal: Dict[str, List[Tuple[int, torch.Tensor]]] = {}
        self._probe_delta_norms: Dict[str, Dict[int, List[Tuple[int, torch.Tensor]]]] = {}
        self._probe_x_norms: Dict[str, Dict[int, List[Tuple[int, torch.Tensor]]]] = {}
        self._probe_deltas: Dict[str, Dict[int, List[Tuple[int, torch.Tensor]]]] = {}

    def record_message(self, site: str, round_idx: int, batch_start: int, latent: torch.Tensor) -> None:
        if not self.trace_messages:
            return
        if site not in MESSAGE_SITES:
            raise ValueError(f"Unsupported role-profile message site: {site!r}")
        self._record_round_tensor(self._messages, site, round_idx, batch_start, latent)

    def record_state(self, role: str, round_idx: int, batch_start: int, latent: torch.Tensor) -> None:
        if not self.trace_states:
            return
        canonical_role = "critic" if role == "refiner" else str(role)
        if canonical_role not in {"planner", "critic", "solver"}:
            raise ValueError(f"Unsupported role-profile role: {role!r}")
        self._record_round_tensor(self._states, canonical_role, round_idx, batch_start, latent)

    def record_terminal(self, name: str, batch_start: int, latent: torch.Tensor) -> None:
        if not self.trace_terminal:
            return
        chunks = self._terminal.setdefault(str(name), [])
        chunks.append((int(batch_start), latent.detach().to(device="cpu", dtype=self.dtype)))

    def record_probe_delta(
        self,
        site: str,
        round_idx: int,
        batch_start: int,
        delta: Optional[torch.Tensor],
        delta_norm: torch.Tensor,
        x_norm: torch.Tensor,
    ) -> None:
        canonical_site = canonical_probe_site(site)
        self._record_round_tensor(
            self._probe_delta_norms,
            canonical_site,
            round_idx,
            batch_start,
            delta_norm.detach().float(),
            dtype=torch.float32,
        )
        self._record_round_tensor(
            self._probe_x_norms,
            canonical_site,
            round_idx,
            batch_start,
            x_norm.detach().float(),
            dtype=torch.float32,
        )
        if delta is not None and self.record_actual_delta:
            self._record_round_tensor(
                self._probe_deltas,
                canonical_site,
                round_idx,
                batch_start,
                delta,
            )

    def _record_round_tensor(
        self,
        store: Dict[str, Dict[int, List[Tuple[int, torch.Tensor]]]],
        key: str,
        round_idx: int,
        batch_start: int,
        latent: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        target_dtype = dtype if dtype is not None else self.dtype
        cpu_latent = latent.detach().to(device="cpu", dtype=target_dtype)
        key_chunks = store.setdefault(str(key), {})
        round_chunks = key_chunks.setdefault(int(round_idx), [])
        round_chunks.append((int(batch_start), cpu_latent))

    @staticmethod
    def _concat_round_chunks(
        store: Dict[str, Dict[int, List[Tuple[int, torch.Tensor]]]]
    ) -> Dict[str, Dict[int, torch.Tensor]]:
        out: Dict[str, Dict[int, torch.Tensor]] = {}
        for key in sorted(store):
            out[key] = {}
            for round_idx, chunks in sorted(store[key].items()):
                if not chunks:
                    continue
                ordered = [chunk for _, chunk in sorted(chunks, key=lambda item: item[0])]
                out[key][int(round_idx)] = torch.cat(ordered, dim=0)
        return out

    @staticmethod
    def _concat_terminal_chunks(store: Dict[str, List[Tuple[int, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        for key in sorted(store):
            chunks = store[key]
            if not chunks:
                continue
            ordered = [chunk for _, chunk in sorted(chunks, key=lambda item: item[0])]
            out[key] = torch.cat(ordered, dim=0)
        return out

    def _build_probe_deltas(self) -> Dict[str, Dict[int, Dict[str, torch.Tensor]]]:
        out: Dict[str, Dict[int, Dict[str, torch.Tensor]]] = {}
        delta_norms = self._concat_round_chunks(self._probe_delta_norms)
        x_norms = self._concat_round_chunks(self._probe_x_norms)
        deltas = self._concat_round_chunks(self._probe_deltas)
        for site, rounds in delta_norms.items():
            out[site] = {}
            for round_idx, delta_norm in rounds.items():
                entry: Dict[str, torch.Tensor] = {"delta_norm": delta_norm}
                if site in x_norms and round_idx in x_norms[site]:
                    entry["x_norm"] = x_norms[site][round_idx]
                if site in deltas and round_idx in deltas[site]:
                    entry["delta"] = deltas[site][round_idx]
                out[site][int(round_idx)] = entry
        return out

    def save(self) -> None:
        out = {
            "metadata": self.metadata,
            "sample_ids": self.sample_ids,
            "sample_indices": self.sample_indices,
            "messages": self._concat_round_chunks(self._messages),
            "states": self._concat_round_chunks(self._states),
            "terminal": self._concat_terminal_chunks(self._terminal),
            "probe_deltas": self._build_probe_deltas(),
        }
        out_dir = os.path.dirname(self.path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        torch.save(out, self.path)
