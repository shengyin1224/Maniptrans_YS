from __future__ import annotations

import os
from typing import Dict

import numpy as np
import torch
import trimesh
from yourdfpy import URDF


def quat_xyzw_to_rotmat(quat_xyzw: torch.Tensor) -> torch.Tensor:
    x = quat_xyzw[..., 0]
    y = quat_xyzw[..., 1]
    z = quat_xyzw[..., 2]
    w = quat_xyzw[..., 3]

    norm = torch.sqrt(x * x + y * y + z * z + w * w).clamp_min(1e-8)
    x = x / norm
    y = y / norm
    z = z / norm
    w = w / norm

    xx = x * x
    yy = y * y
    zz = z * z
    ww = w * w
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    rot = torch.stack(
        [
            ww + xx - yy - zz,
            2.0 * (xy - wz),
            2.0 * (xz + wy),
            2.0 * (xy + wz),
            ww - xx + yy - zz,
            2.0 * (yz - wx),
            2.0 * (xz - wy),
            2.0 * (yz + wx),
            ww - xx - yy + zz,
        ],
        dim=-1,
    )
    return rot.reshape(quat_xyzw.shape[:-1] + (3, 3))


def _ensure_trimesh(geometry) -> trimesh.Trimesh:
    if isinstance(geometry, trimesh.Trimesh):
        return geometry
    if isinstance(geometry, trimesh.Scene):
        return _ensure_trimesh(geometry.to_geometry())
    if isinstance(geometry, (list, tuple)):
        meshes = [g for g in geometry if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("No trimesh geometry found in iterable geometry container")
        return trimesh.util.concatenate(meshes)
    if hasattr(geometry, "geometry"):
        geometries = list(getattr(geometry, "geometry").values())
        meshes = [g for g in geometries if isinstance(g, trimesh.Trimesh)]
        if not meshes:
            raise ValueError("No trimesh geometry found in scene-like geometry container")
        return trimesh.util.concatenate(meshes)
    raise TypeError(f"Unsupported geometry type: {type(geometry)}")


def load_urdf_sample_points(
    urdf_path: str,
    num_points: int = 32,
    use_collision_mesh: bool = False,
    seed: int = 2024,
) -> np.ndarray:
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF file not found: {urdf_path}")

    urdf = URDF.load(
        urdf_path,
        build_scene_graph=True,
        build_collision_scene_graph=use_collision_mesh,
        load_meshes=True,
        load_collision_meshes=use_collision_mesh,
        force_mesh=True,
        force_collision_mesh=True,
    )

    scene = urdf.collision_scene if use_collision_mesh and urdf.collision_scene is not None else urdf.scene
    if scene is None:
        raise RuntimeError(f"Failed to build scene from URDF: {urdf_path}")

    mesh = _ensure_trimesh(scene.to_geometry())
    if mesh.vertices.shape[0] == 0:
        raise RuntimeError(f"Mesh extracted from URDF has no vertices: {urdf_path}")

    try:
        points, _ = trimesh.sample.sample_surface_even(mesh, count=num_points, seed=seed)
    except Exception:
        points, _ = trimesh.sample.sample_surface(mesh, count=num_points)

    if points.shape[0] == 0:
        raise RuntimeError(f"Failed to sample any points from URDF mesh: {urdf_path}")

    if points.shape[0] < num_points:
        repeat = int(np.ceil(float(num_points) / float(points.shape[0])))
        points = np.tile(points, (repeat, 1))

    return np.asarray(points[:num_points], dtype=np.float32)


def compute_sampled_object_pose_errors(
    current_pos: torch.Tensor,
    current_quat_xyzw: torch.Tensor,
    target_pos: torch.Tensor,
    target_quat_xyzw: torch.Tensor,
    sample_points_local: torch.Tensor,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    num_envs = current_pos.shape[0]
    num_objs = current_pos.shape[1]
    num_points = sample_points_local.shape[2]

    flat_curr_quat = current_quat_xyzw.reshape(-1, 4)
    flat_targ_quat = target_quat_xyzw.reshape(-1, 4)
    flat_curr_rot = quat_xyzw_to_rotmat(flat_curr_quat)
    flat_targ_rot = quat_xyzw_to_rotmat(flat_targ_quat)

    flat_points_local = sample_points_local.reshape(-1, num_points, 3)
    flat_curr_pos = current_pos.reshape(-1, 3)
    flat_targ_pos = target_pos.reshape(-1, 3)

    curr_world = torch.bmm(flat_curr_rot, flat_points_local.transpose(1, 2)).transpose(1, 2)
    curr_world = curr_world + flat_curr_pos.unsqueeze(1)
    targ_world = torch.bmm(flat_targ_rot, flat_points_local.transpose(1, 2)).transpose(1, 2)
    targ_world = targ_world + flat_targ_pos.unsqueeze(1)

    point_disp = torch.norm(curr_world - targ_world, dim=-1)
    mean_point_err = point_disp.mean(dim=-1)

    curr_centroid = curr_world.mean(dim=1)
    targ_centroid = targ_world.mean(dim=1)
    centroid_err = torch.norm(curr_centroid - targ_centroid, dim=-1)

    curr_centered = curr_world - curr_centroid.unsqueeze(1)
    targ_centered = targ_world - targ_centroid.unsqueeze(1)
    centered_disp = torch.norm(curr_centered - targ_centered, dim=-1)
    mean_centered_disp = centered_disp.mean(dim=-1)

    local_centroid = flat_points_local.mean(dim=1)
    local_radius = torch.norm(flat_points_local - local_centroid.unsqueeze(1), dim=-1).mean(dim=-1)
    local_radius = torch.clamp(local_radius, min=eps)
    rot_err_rad = mean_centered_disp / local_radius
    rot_err_deg = rot_err_rad * (180.0 / np.pi)

    return {
        "mean_point_err": mean_point_err.reshape(num_envs, num_objs),
        "centroid_err": centroid_err.reshape(num_envs, num_objs),
        "mean_centered_disp": mean_centered_disp.reshape(num_envs, num_objs),
        "rot_err_rad": rot_err_rad.reshape(num_envs, num_objs),
        "rot_err_deg": rot_err_deg.reshape(num_envs, num_objs),
        "local_radius": local_radius.reshape(num_envs, num_objs),
    }
