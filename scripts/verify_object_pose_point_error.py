from __future__ import annotations

import argparse
import importlib.util
import math
import pathlib

import torch


def _load_pose_error_utils():
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    module_path = repo_root / "maniptrans_envs" / "lib" / "utils" / "object_pose_error_utils.py"
    spec = importlib.util.spec_from_file_location("object_pose_error_utils", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_pose_error_utils = _load_pose_error_utils()
compute_sampled_object_pose_errors = _pose_error_utils.compute_sampled_object_pose_errors
load_urdf_sample_points = _pose_error_utils.load_urdf_sample_points


def axis_angle_to_xyzw(axis: torch.Tensor, angle_rad: float) -> torch.Tensor:
    axis = axis / torch.clamp(torch.norm(axis), min=1e-8)
    half = angle_rad * 0.5
    sin_half = math.sin(half)
    return torch.tensor(
        [axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, math.cos(half)],
        dtype=torch.float32,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=str, default="")
    parser.add_argument("--num-points", type=int, default=32)
    args = parser.parse_args()

    if args.urdf:
        pts = load_urdf_sample_points(args.urdf, num_points=args.num_points)
        sample_points = torch.from_numpy(pts).unsqueeze(0).unsqueeze(0)
        source = args.urdf
    else:
        sample_points = torch.tensor(
            [
                [-0.10, -0.02, -0.02],
                [-0.10, -0.02, 0.02],
                [-0.10, 0.02, -0.02],
                [-0.10, 0.02, 0.02],
                [0.10, -0.02, -0.02],
                [0.10, -0.02, 0.02],
                [0.10, 0.02, -0.02],
                [0.10, 0.02, 0.02],
            ],
            dtype=torch.float32,
        ).unsqueeze(0).unsqueeze(0)
        source = "synthetic_box"

    base_pos = torch.zeros(1, 1, 3, dtype=torch.float32)
    base_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32).view(1, 1, 4)

    trans_delta = torch.tensor([0.001, -0.0005, 0.0], dtype=torch.float32).view(1, 1, 3)
    trans_errors = compute_sampled_object_pose_errors(
        current_pos=base_pos + trans_delta,
        current_quat_xyzw=base_quat,
        target_pos=base_pos,
        target_quat_xyzw=base_quat,
        sample_points_local=sample_points,
    )
    expected_trans = torch.norm(trans_delta[0, 0]).item()

    rot_quat = axis_angle_to_xyzw(torch.tensor([0.0, 0.0, 1.0]), math.radians(1.0)).view(1, 1, 4)
    rot_errors = compute_sampled_object_pose_errors(
        current_pos=base_pos,
        current_quat_xyzw=rot_quat,
        target_pos=base_pos,
        target_quat_xyzw=base_quat,
        sample_points_local=sample_points,
    )

    print(f"source={source}")
    print(
        "translation_case "
        f"expected_mean_point_err={expected_trans:.8f} "
        f"got_mean_point_err={trans_errors['mean_point_err'][0, 0].item():.8f} "
        f"got_centroid_err={trans_errors['centroid_err'][0, 0].item():.8f} "
        f"got_rot_err_deg={trans_errors['rot_err_deg'][0, 0].item():.8f}"
    )
    print(
        "rotation_case "
        f"input_angle_deg=1.00000000 "
        f"got_mean_point_err={rot_errors['mean_point_err'][0, 0].item():.8f} "
        f"got_centroid_err={rot_errors['centroid_err'][0, 0].item():.8f} "
        f"got_rot_err_deg={rot_errors['rot_err_deg'][0, 0].item():.8f}"
    )


if __name__ == "__main__":
    main()
