from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is expected for ManiSkill3 usage.
    torch = None


class PraxisRemoteInference:
    """Thin SIMPLER wrapper around the Praxis gRPC policy client."""

    def __init__(
        self,
        *,
        host: str = "localhost",
        port: int = 50051,
        policy_setup: str = "widowx_bridge",
        action_scale: float = 1.0,
        primary_image_key: str = "observation.images.image",
        additional_image_keys: Mapping[str, str] | None = None,
        state_key: str = "observation.state",
        task_key: str = "task",
        policy_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        try:
            from praxis_remote import PolicyClient
        except ImportError as exc:  # pragma: no cover - import failure is user setup.
            raise ImportError(
                "PraxisRemoteInference requires praxis_remote to be importable. "
                "Install the praxis-remote package before using this wrapper."
            ) from exc

        self.client = PolicyClient(host=host, port=int(port))
        self.policy_setup = policy_setup
        self.action_scale = float(action_scale)
        self.state_key = state_key
        self.task_key = task_key
        self.image_key_map = {"image": primary_image_key}
        if additional_image_keys is not None:
            self.image_key_map.update(
                {
                    str(local_key): str(remote_key)
                    for local_key, remote_key in additional_image_keys.items()
                }
            )
        self.policy_kwargs = dict(policy_kwargs or {})
        self.task = None
        self._task_signature: str | tuple[str, ...] | None = None

    def health_check(self) -> tuple[bool, str]:
        return self.client.health_check()

    def close(self) -> None:
        self.client.close()

    def reset(self, task_description: str | Sequence[str] | None = None) -> None:
        self.client.reset()
        self.task = task_description
        self._task_signature = _normalize_task_description(task_description)

    def step(
        self,
        image: np.ndarray | Any,
        task_description: str | Sequence[str] | None = None,
        *args,
        state: np.ndarray | Any | None = None,
        images: Mapping[str, np.ndarray | Any] | None = None,
        policy_kwargs: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> tuple[dict[str, np.ndarray | Any], dict[str, np.ndarray | Any]]:
        """Run one policy step.

        ``image`` matches the current SIMPLER call sites: one primary image for
        ManiSkill2 or a batch of primary images for ManiSkill3. Extra Praxis image
        observations can be provided through ``images`` and are mapped by name
        through ``additional_image_keys``.
        """
        del args, kwargs

        task_signature = _normalize_task_description(task_description)
        if task_signature is not None and task_signature != self._task_signature:
            self.reset(task_description)

        observations, batched_input, return_device = self._build_observations(
            image=image,
            state=state,
            images=images,
            task_description=task_description,
        )
        request_kwargs = dict(self.policy_kwargs)
        if policy_kwargs is not None:
            request_kwargs.update(policy_kwargs)
        action = np.asarray(
            self.client.predict_observations(
                observations,
                policy_kwargs=request_kwargs,
            ),
            dtype=np.float32,
        )
        if batched_input and action.ndim == 1:
            action = action[None, :]

        raw_action, env_action = self._format_action(
            action,
            return_device=return_device,
        )
        self.task = task_description
        self._task_signature = task_signature
        return raw_action, env_action

    def visualize_epoch(
        self,
        predicted_raw_actions: Sequence[Mapping[str, np.ndarray | Any]],
        images: Sequence[np.ndarray | Any],
        save_path: str,
    ) -> None:
        import matplotlib.pyplot as plt

        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        action_rows = [
            _to_numpy(raw_action["action"]).astype(np.float32, copy=False).reshape(-1)
            for raw_action in predicted_raw_actions
        ]
        action_traj = (
            np.stack(action_rows, axis=0)
            if action_rows
            else np.zeros((0, 7), dtype=np.float32)
        )
        preview = (
            _to_numpy(images[-1]) if images else np.zeros((32, 32, 3), dtype=np.uint8)
        )
        if preview.ndim == 4:
            preview = preview[0]
        if (
            preview.ndim == 3
            and preview.shape[0] in {1, 3, 4}
            and preview.shape[-1] not in {3, 4}
        ):
            preview = np.moveaxis(preview, 0, -1)
        if preview.ndim == 3 and preview.shape[-1] == 4:
            preview = preview[..., :3]
        if preview.dtype != np.uint8:
            preview = np.clip(preview * 255.0, 0, 255).astype(np.uint8)

        figure, axes = plt.subplots(2, 1, figsize=(12, 6))
        axes[0].imshow(preview)
        axes[0].axis("off")
        axes[0].set_title("Final Observation")
        if action_traj.size > 0:
            axes[1].plot(action_traj)
        axes[1].set_title("Praxis Action Trajectory")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("Action Value")
        figure.tight_layout()
        figure.savefig(save_path)
        plt.close(figure)

    def _build_observations(
        self,
        *,
        image: np.ndarray | Any,
        state: np.ndarray | Any | None,
        images: Mapping[str, np.ndarray | Any] | None,
        task_description: str | Sequence[str] | None,
    ) -> tuple[list[dict[str, np.ndarray | str]], bool, Any]:
        prepared_images = {
            self.image_key_map["image"]: _prepare_image_array(image),
        }
        if images is not None:
            for local_key, value in images.items():
                remote_key = self.image_key_map.get(str(local_key), str(local_key))
                prepared_images[remote_key] = _prepare_image_array(value)

        prepared_state = _prepare_state_array(state) if state is not None else None

        batch_size = 1
        batched_input = False
        return_device = _resolve_torch_device(image)
        if return_device is None:
            return_device = _resolve_torch_device(state)
        if images is not None and return_device is None:
            for value in images.values():
                return_device = _resolve_torch_device(value)
                if return_device is not None:
                    break

        for key, value in prepared_images.items():
            current_batch_size, current_is_batched = _infer_batch_size_from_array(
                value,
                key=key,
                single_rank=3,
                batched_rank=4,
            )
            batch_size, batched_input = _merge_batch_shape(
                batch_size=batch_size,
                batched_input=batched_input,
                current_batch_size=current_batch_size,
                current_is_batched=current_is_batched,
                name=key,
            )

        if prepared_state is not None:
            current_batch_size, current_is_batched = _infer_batch_size_from_state(
                prepared_state
            )
            batch_size, batched_input = _merge_batch_shape(
                batch_size=batch_size,
                batched_input=batched_input,
                current_batch_size=current_batch_size,
                current_is_batched=current_is_batched,
                name=self.state_key,
            )

        if task_description is not None and not isinstance(task_description, str):
            current_batch_size = len(task_description)
            batch_size, batched_input = _merge_batch_shape(
                batch_size=batch_size,
                batched_input=batched_input,
                current_batch_size=current_batch_size,
                current_is_batched=True,
                name=self.task_key,
            )

        observations: list[dict[str, np.ndarray | str]] = [
            {} for _ in range(batch_size)
        ]
        for key, value in prepared_images.items():
            _assign_array_batch(observations, key=key, value=value)
        if prepared_state is not None:
            _assign_array_batch(observations, key=self.state_key, value=prepared_state)
        _assign_task_batch(
            observations,
            task_key=self.task_key,
            task_description=task_description,
        )

        return observations, batched_input, return_device

    def _format_action(
        self,
        action: np.ndarray,
        *,
        return_device: Any,
    ) -> tuple[dict[str, np.ndarray | Any], dict[str, np.ndarray | Any]]:
        action = np.asarray(action, dtype=np.float32)
        if action.ndim not in {1, 2} or action.shape[-1] != 7:
            raise ValueError(
                f"Expected Bridge action shape (7,) or (B, 7), got {action.shape}."
            )
        if action.ndim == 1:
            world_vector = action[:3] * self.action_scale
            rotation_delta = action[3:6]
            gripper = action[6:7]
            terminate_episode = np.zeros((1,), dtype=np.float32)
        else:
            world_vector = action[:, :3] * self.action_scale
            rotation_delta = action[:, 3:6]
            gripper = action[:, 6:7]
            terminate_episode = np.zeros((action.shape[0], 1), dtype=np.float32)

        rotation_delta = rotation_delta.astype(np.float32, copy=False)
        rotation_delta = rotation_delta * self.action_scale

        if self.policy_setup == "widowx_bridge":
            gripper = np.where(gripper > 0.5, 1.0, -1.0).astype(
                np.float32,
                copy=False,
            )

        raw_action = {
            "action": _maybe_to_torch(action, return_device=return_device),
            "world_vector": _maybe_to_torch(world_vector, return_device=return_device),
            "rotation_delta": _maybe_to_torch(
                rotation_delta,
                return_device=return_device,
            ),
            "gripper": _maybe_to_torch(gripper, return_device=return_device),
        }
        env_action = {
            "world_vector": _maybe_to_torch(world_vector, return_device=return_device),
            "rotation_delta": _maybe_to_torch(
                rotation_delta,
                return_device=return_device,
            ),
            "gripper": _maybe_to_torch(gripper, return_device=return_device),
            "terminate_episode": _maybe_to_torch(
                terminate_episode,
                return_device=return_device,
            ),
        }
        return raw_action, env_action


def _prepare_image_array(image: np.ndarray | Any) -> np.ndarray:
    array = _to_numpy(image)
    if array.ndim == 3:
        array = _image_to_chw(array)
    elif array.ndim == 4:
        if array.shape[-1] in {3, 4}:
            array = array[..., :3]
            array = np.moveaxis(array, -1, 1)
        elif array.shape[1] in {3, 4}:
            array = array[:, :3]
        else:
            raise ValueError(
                f"Unsupported batched image shape {array.shape}; expected BHWC or BCHW."
            )
    else:
        raise ValueError(
            f"Unsupported image rank {array.ndim}; expected HWC/BHWC or CHW/BCHW."
        )

    if np.issubdtype(array.dtype, np.integer):
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
    else:
        array = array.astype(np.float32, copy=False)
    return np.ascontiguousarray(array)


def _image_to_chw(image: np.ndarray) -> np.ndarray:
    if image.shape[-1] in {3, 4}:
        image = image[..., :3]
        return np.moveaxis(image, -1, 0)
    if image.shape[0] in {3, 4}:
        return image[:3]
    raise ValueError(
        f"Unsupported image shape {image.shape}; expected HWC or CHW with 3 channels."
    )


def _prepare_state_array(state: np.ndarray | Any) -> np.ndarray:
    array = _to_numpy(state).astype(np.float32, copy=False)
    if array.ndim not in {1, 2}:
        raise ValueError(
            f"Unsupported state shape {array.shape}; expected (D,) or (B, D)."
        )
    return np.ascontiguousarray(array)


def _assign_array_batch(
    observations: list[dict[str, np.ndarray | str]],
    *,
    key: str,
    value: np.ndarray,
) -> None:
    if value.ndim in {1, 3}:
        if len(observations) != 1:
            raise ValueError(
                f"Observation key {key!r} is unbatched but the request batch size is {len(observations)}."
            )
        observations[0][key] = value
        return
    if value.ndim in {2, 4}:
        if int(value.shape[0]) != len(observations):
            raise ValueError(
                f"Observation key {key!r} has batch size {value.shape[0]}, "
                f"expected {len(observations)}."
            )
        for index in range(len(observations)):
            observations[index][key] = value[index]
        return
    raise ValueError(
        f"Unsupported observation tensor rank {value.ndim} for key {key!r}."
    )


def _assign_task_batch(
    observations: list[dict[str, np.ndarray | str]],
    *,
    task_key: str,
    task_description: str | Sequence[str] | None,
) -> None:
    if task_description is None:
        return
    if isinstance(task_description, str):
        if len(observations) != 1:
            raise ValueError(
                f"Task description is unbatched but the request batch size is {len(observations)}."
            )
        observations[0][task_key] = task_description
        return
    if len(task_description) != len(observations):
        raise ValueError(
            f"Task batch has length {len(task_description)}, expected {len(observations)}."
        )
    for index, value in enumerate(task_description):
        observations[index][task_key] = str(value)


def _infer_batch_size_from_array(
    value: np.ndarray,
    *,
    key: str,
    single_rank: int,
    batched_rank: int,
) -> tuple[int, bool]:
    if value.ndim == single_rank:
        return 1, False
    if value.ndim == batched_rank:
        return int(value.shape[0]), True
    raise ValueError(
        f"Observation key {key!r} has rank {value.ndim}; "
        f"expected {single_rank} or {batched_rank}."
    )


def _infer_batch_size_from_state(value: np.ndarray) -> tuple[int, bool]:
    if value.ndim == 1:
        return 1, False
    if value.ndim == 2:
        return int(value.shape[0]), True
    raise ValueError(
        f"Unsupported state rank {value.ndim}; expected 1 or 2 dimensions."
    )


def _merge_batch_shape(
    *,
    batch_size: int,
    batched_input: bool,
    current_batch_size: int,
    current_is_batched: bool,
    name: str,
) -> tuple[int, bool]:
    if current_is_batched:
        if batch_size != 1 and batch_size != current_batch_size:
            raise ValueError(
                f"Batched input for {name!r} has size {current_batch_size}, "
                f"expected {batch_size}."
            )
        return current_batch_size, True
    return batch_size, batched_input


def _normalize_task_description(
    task_description: str | Sequence[str] | None,
) -> str | tuple[str, ...] | None:
    if task_description is None:
        return None
    if isinstance(task_description, str):
        return task_description
    return tuple(str(item) for item in task_description)


def _to_numpy(value: np.ndarray | Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _is_torch_value(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _resolve_torch_device(value: Any):
    if _is_torch_value(value):
        return value.device
    return None


def _maybe_to_torch(
    array: np.ndarray | Any,
    *,
    return_device: Any,
) -> np.ndarray | Any:
    if return_device is None:
        return array
    if torch is None:  # pragma: no cover - guarded by return_device construction.
        raise RuntimeError("torch is required to return torch-based Praxis actions.")
    return torch.from_numpy(np.asarray(array)).to(device=return_device)
