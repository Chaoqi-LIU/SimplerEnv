from __future__ import annotations

from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - torch is expected for ManiSkill3 usage.
    torch = None


def normalize_bridge_gripper_qpos(qpos: Any, qlimits: Any):
    """Return Bridge-style gripper opening from two finger joint positions.

    Bridge state stores gripper opening on roughly a 0..1 scale, not raw joint
    meters. ManiSkill's table tasks use the same ``sum(qpos[-2:]) / max_width``
    convention for gripper opening rewards.
    """
    if torch is None:  # pragma: no cover - guarded by Simpler ManiSkill runtime.
        raise RuntimeError("torch is required to normalize Bridge gripper qpos.")

    qpos_tensor = qpos if torch.is_tensor(qpos) else torch.as_tensor(qpos)
    if qpos_tensor.ndim == 1:
        qpos_tensor = qpos_tensor[None, :]
    if qpos_tensor.ndim != 2 or qpos_tensor.shape[-1] < 2:
        raise ValueError(
            f"Expected qpos shape (B, D>=2) or (D>=2,), got {tuple(qpos_tensor.shape)}"
        )

    qlimits_tensor = (
        qlimits if torch.is_tensor(qlimits) else torch.as_tensor(qlimits)
    ).to(device=qpos_tensor.device, dtype=qpos_tensor.dtype)
    if qlimits_tensor.ndim == 2:
        qlimits_tensor = qlimits_tensor[None, :, :]
    if (
        qlimits_tensor.ndim != 3
        or qlimits_tensor.shape[-2] < 2
        or qlimits_tensor.shape[-1] != 2
    ):
        raise ValueError(
            "Expected qlimits shape (B, D>=2, 2) or (D>=2, 2), "
            f"got {tuple(qlimits_tensor.shape)}"
        )
    if qlimits_tensor.shape[0] == 1 and qpos_tensor.shape[0] != 1:
        qlimits_tensor = qlimits_tensor.expand(qpos_tensor.shape[0], -1, -1)
    if qlimits_tensor.shape[0] != qpos_tensor.shape[0]:
        raise ValueError(
            f"qpos batch size {qpos_tensor.shape[0]} does not match qlimits "
            f"batch size {qlimits_tensor.shape[0]}"
        )

    max_width = qlimits_tensor[:, -2:, 1].sum(dim=-1, keepdim=True)
    return qpos_tensor[:, -2:].sum(dim=-1, keepdim=True) / max_width.clamp_min(1e-6)
