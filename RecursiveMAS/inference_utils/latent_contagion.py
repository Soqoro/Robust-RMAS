from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Optional

import torch


@dataclass
class PerturbConfig:
    mode: str = "none"
    site: str = ""
    epsilon: float = 0.0
    round_idx: int = 0
    seed: int = 42
    enabled: bool = False


def stable_seed(base_seed: int, site: str, round_idx: int, batch_start: int) -> int:
    payload = f"latent-contagion:{int(base_seed)}:{site}:{int(round_idx)}:{int(batch_start)}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**63)


def _per_sample_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(x.reshape(x.size(0), -1), dim=1)


def normalized_random_perturbation(x: torch.Tensor, seed: int) -> torch.Tensor:
    if x.dim() not in (2, 3):
        raise ValueError(f"Expected latent tensor with shape [B,D] or [B,T,D], got {tuple(x.shape)}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    noise = torch.randn(tuple(x.shape), generator=generator, dtype=torch.float32, device="cpu")
    noise = noise.to(device=x.device)
    norm = _per_sample_norm(noise).clamp_min(1e-12)
    view_shape = (x.size(0),) + (1,) * (x.dim() - 1)
    return noise / norm.view(view_shape)


def maybe_perturb(
    x: torch.Tensor,
    cfg: Optional[PerturbConfig],
    site: str,
    round_idx: int,
    batch_start: int,
) -> tuple[torch.Tensor, Optional[dict]]:
    if cfg is None or not cfg.enabled:
        return x, None
    if cfg.site != site or float(cfg.epsilon) <= 0.0:
        return x, None
    if cfg.mode == "one_shot":
        if int(cfg.round_idx) != int(round_idx):
            return x, None
    elif cfg.mode != "persistent":
        return x, None

    seed = stable_seed(cfg.seed, site, round_idx, batch_start)
    direction = normalized_random_perturbation(x, seed)
    x_float = x.detach().float()
    x_norm = _per_sample_norm(x_float)
    view_shape = (x.size(0),) + (1,) * (x.dim() - 1)
    delta_float = float(cfg.epsilon) * x_norm.view(view_shape) * direction
    x_tilde = (x_float + delta_float).to(dtype=x.dtype)

    applied_delta = x_tilde.detach().float() - x_float
    delta_norm = _per_sample_norm(applied_delta)
    relative_delta_norm = torch.where(
        x_norm > 0,
        delta_norm / x_norm.clamp_min(1e-12),
        torch.zeros_like(delta_norm),
    )
    meta = {
        "mode": cfg.mode,
        "applied": True,
        "site": site,
        "round_idx": int(round_idx),
        "epsilon": float(cfg.epsilon),
        "x_norm_mean": float(x_norm.mean().item()) if x_norm.numel() else 0.0,
        "delta_norm_mean": float(delta_norm.mean().item()) if delta_norm.numel() else 0.0,
        "relative_delta_norm_mean": (
            float(relative_delta_norm.mean().item()) if relative_delta_norm.numel() else 0.0
        ),
        "batch_start": int(batch_start),
    }
    return x_tilde, meta
