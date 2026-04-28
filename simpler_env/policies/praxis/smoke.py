from __future__ import annotations

import argparse

import numpy as np

try:
    from simpler_env.policies.praxis.remote_model import PraxisRemoteInference
except ModuleNotFoundError:
    from remote_model import PraxisRemoteInference


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke-test the Praxis remote SIMPLER wrapper against a running Praxis server."
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument(
        "--primary-image-key",
        default="observation.images.primary",
    )
    parser.add_argument(
        "--wrist-image-key",
        default="observation.images.wrist",
    )
    parser.add_argument("--task", default="smoke task")
    parser.add_argument("--state-dim", type=int, default=4)
    args = parser.parse_args()

    model = PraxisRemoteInference(
        host=args.host,
        port=args.port,
        primary_image_key=args.primary_image_key,
        additional_image_keys={"wrist": args.wrist_image_key},
    )
    ready, info = model.health_check()
    print(f"health_check ready={ready} info={info}")

    primary = np.full((2, 2, 3), 64, dtype=np.uint8)
    wrist = np.full((2, 2, 3), 128, dtype=np.uint8)
    state = np.arange(args.state_dim, dtype=np.float32)

    model.reset(args.task)
    raw_action, action = model.step(
        primary,
        args.task,
        state=state,
        images={"wrist": wrist},
    )

    print("raw_action keys:", sorted(raw_action.keys()))
    print("action keys:", sorted(action.keys()))
    print("raw action:", np.asarray(raw_action["action"]).tolist())
    print("world_vector:", np.asarray(action["world_vector"]).tolist())
    print("rotation_delta:", np.asarray(action["rotation_delta"]).tolist())
    print("gripper:", np.asarray(action["gripper"]).tolist())
    print("terminate_episode:", np.asarray(action["terminate_episode"]).tolist())

    model.close()


if __name__ == "__main__":
    main()
