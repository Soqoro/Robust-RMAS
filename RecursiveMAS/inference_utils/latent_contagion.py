from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Optional

import torch


@dataclass
class PerturbConfig:
    mode: str = "none"
    site: str = ""
    epsilon: float = 0.0
    round_idx: int = 0
    seed: int = 42
    enabled: bool = False
    direction: str = "random"
    steering_bank_path: str = ""
    steering_method: str = ""
    steering_id: str = ""
    steering_bank: Optional[dict] = None


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


def _lookup_round_key(round_map: Any, round_idx: int, site: str) -> Any:
    if not isinstance(round_map, dict):
        raise ValueError(f"Steering bank entry for site={site!r} must be a dict of rounds.")
    if round_idx in round_map:
        return round_map[round_idx]
    round_key = str(round_idx)
    if round_key in round_map:
        return round_map[round_key]
    available = ", ".join(str(key) for key in sorted(round_map, key=str))
    raise ValueError(
        f"Steering bank has no direction for site={site!r} round_idx={round_idx}. "
        f"Available rounds: [{available}]"
    )


def _bank_direction(x: torch.Tensor, cfg: PerturbConfig, site: str, round_idx: int) -> torch.Tensor:
    if cfg.steering_bank is None:
        raise ValueError("lc_direction='bank' requires a loaded steering_bank.")
    directions = cfg.steering_bank.get("directions")
    if not isinstance(directions, dict):
        raise ValueError("Steering bank is missing a 'directions' dict.")
    if cfg.steering_method not in directions:
        available = ", ".join(str(key) for key in sorted(directions, key=str))
        raise ValueError(
            f"Steering bank has no method={cfg.steering_method!r}. "
            f"Available methods: [{available}]"
        )
    method_bank = directions[cfg.steering_method]
    if not isinstance(method_bank, dict) or site not in method_bank:
        available = ", ".join(str(key) for key in sorted(method_bank, key=str)) if isinstance(method_bank, dict) else ""
        raise ValueError(
            f"Steering bank has no site={site!r} for method={cfg.steering_method!r}. "
            f"Available sites: [{available}]"
        )

    direction = _lookup_round_key(method_bank[site], int(round_idx), site)
    if not isinstance(direction, torch.Tensor):
        raise ValueError(
            f"Steering direction for method={cfg.steering_method!r} "
            f"site={site!r} round_idx={round_idx} must be a tensor."
        )
    direction = direction.detach().to(device=x.device, dtype=torch.float32)
    direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-12)

    if x.dim() == 3 and direction.dim() == 2:
        if tuple(direction.shape) != tuple(x.shape[1:]):
            raise ValueError(
                f"Steering direction shape {tuple(direction.shape)} does not match "
                f"latent shape without batch {tuple(x.shape[1:])} for site={site!r} "
                f"round_idx={round_idx}."
            )
        return direction.unsqueeze(0).expand(x.size(0), -1, -1)

    if x.dim() == 2 and direction.dim() == 1:
        if int(direction.size(0)) != int(x.size(1)):
            raise ValueError(
                f"Steering direction shape {tuple(direction.shape)} does not match "
                f"latent shape without batch {tuple(x.shape[1:])} for site={site!r} "
                f"round_idx={round_idx}."
            )
        return direction.unsqueeze(0).expand(x.size(0), -1)

    raise ValueError(
        f"Unsupported steering direction shape {tuple(direction.shape)} for latent shape "
        f"{tuple(x.shape)} at site={site!r} round_idx={round_idx}. Expected [T,D] for "
        "[B,T,D] latents or [D] for [B,D] latents."
    )


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

    direction_name = str(cfg.direction or "random")
    if direction_name == "random":
        seed = stable_seed(cfg.seed, site, round_idx, batch_start)
        direction = normalized_random_perturbation(x, seed)
    elif direction_name == "bank":
        direction = _bank_direction(x, cfg, site, round_idx)
    else:
        raise ValueError(f"Unsupported latent perturbation direction: {direction_name!r}")

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
        "direction": direction_name,
        "steering_method": cfg.steering_method,
        "steering_id": cfg.steering_id,
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
