#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

SCRIPT_DIR = Path(__file__).resolve().parent
for candidate in (
    SCRIPT_DIR,
    Path("/autodl-fs/data/mingyu/Mujoco/projects/TableTennisTrack/scripts"),
    Path("/root/autodl-tmp/mingyu/Mujoco/projects/TableTennisTrack/scripts"),
):
    if candidate.exists() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

import mujoco
import numpy as np

import table_tennis_track as base

BALL_RADIUS = float(getattr(base, "BALL_RADIUS", 0.02))
WALL_X = float(getattr(base, "WALL_X", 1.5))
WALL_HEIGHT = float(getattr(base, "WALL_HEIGHT", 1.52))
MAX_REPLAY_POS_ERR_M = 1e-8
MAX_PADDLE_POS_ERR_M = 1e-8
MAX_CONTACT_PENETRATION_M = 0.03
MAX_GROUND_PENETRATION_M = 0.04
MAX_SPEED_M_S = 80.0
MAX_ANG_SPEED_RAD_S = 3000.0


def _scalar(x):
    try:
        return x.item()
    except Exception:
        return x


def _json_load_scalar(z, key, default=None):
    if key not in z:
        return default
    raw = _scalar(z[key])
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", "replace")
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def _safe_list(x):
    try:
        return np.asarray(x, dtype=float).tolist()
    except Exception:
        return x


def _load_assets(npz_path: Path, metadata: dict):
    assets = {}
    texture_dirs = []
    repo_root = SCRIPT_DIR.parent
    texture_dirs.append(repo_root / "assets" / "textures")
    try:
        base.ensure_visual_assets()
        texture_dirs.append(base.TEXTURE_DIR)
    except Exception:
        pass
    texture_dirs.append(npz_path.parent / "assets" / "textures")
    for tex_dir in texture_dirs:
        if tex_dir.exists():
            for tex in tex_dir.glob("*.png"):
                assets.setdefault(tex.name, tex.read_bytes())
    ep_id = int(metadata.get("episode_id", int(npz_path.stem.split("_")[-1])))
    mesh_dirs = []
    if metadata.get("episode_mesh_dir"):
        mesh_dirs.append(Path(metadata["episode_mesh_dir"]))
    mesh_dirs.append(npz_path.parent / f"episode_{ep_id:06d}_meshes")
    for mesh_dir in mesh_dirs:
        if mesh_dir.exists():
            for mesh in mesh_dir.glob("*.obj"):
                assets[mesh.name] = mesh.read_bytes()
            manifest = mesh_dir / "mesh_manifest.json"
            if manifest.exists():
                assets[manifest.name] = manifest.read_bytes()
            break
    missing = [name for name in ("sketchfab_paddle_mujoco_visual.obj", "sketchfab_ball_mujoco_visual.obj") if name not in assets]
    if missing:
        raise FileNotFoundError(f"missing replay mesh assets for {npz_path}: {missing}")
    return assets


def _quat_err(q1, q2):
    a = np.asarray(q1, dtype=float)
    b = np.asarray(q2, dtype=float)
    if a.shape != b.shape:
        return float("inf")
    a = a / max(np.linalg.norm(a), 1e-12)
    b = b / max(np.linalg.norm(b), 1e-12)
    return float(min(np.max(np.abs(a - b)), np.max(np.abs(a + b))))


def _validation_path(npz_path: Path):
    # Current demo3 archive layout is flat, so sidecar names include the episode id.
    # Episode-directory layouts can still use the literal state_replay_validation.json name.
    if npz_path.parent.name == npz_path.stem:
        return npz_path.parent / "state_replay_validation.json"
    return npz_path.with_name(f"{npz_path.stem}_state_replay_validation.json")


def verify_episode(npz_path: Path, indices=0):
    failure_reasons = []
    report = {
        "schema": "state_replay_validation_v1",
        "validator": "demo3_mujoco_state_replay_validator_v2",
        "engine": "mujoco",
        "episode_npz": str(npz_path),
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "state_sequence_importable": False,
        "engine_replay_verified": False,
        "trajectory_error": {},
        "contact_error": {},
        "penetration": {},
        "task_constraints": {},
        "validation_pass": False,
        "failure_reasons": failure_reasons,
    }
    try:
        z = np.load(npz_path, allow_pickle=True)
        metadata = _json_load_scalar(z, "metadata", {}) or {}
        xml = str(_scalar(z["model_xml"]))
        model = mujoco.MjModel.from_xml_string(xml, assets=_load_assets(npz_path, metadata))
        data = mujoco.MjData(model)
        state_qpos = np.asarray(z["state_qpos"], dtype=float)
        state_qvel = np.asarray(z["state_qvel"], dtype=float)
        state_mocap_pos = np.asarray(z["state_mocap_pos"], dtype=float)
        state_mocap_quat = np.asarray(z["state_mocap_quat"], dtype=float)
        ball_trajectory = np.asarray(z["ball_trajectory"], dtype=float)
        paddle_trajectory = np.asarray(z["paddle_trajectory"], dtype=float)
        state_time = np.asarray(z["state_time"], dtype=float)
        n = int(state_qpos.shape[0])
        if n <= 0:
            failure_reasons.append("no recorded state samples")
        for name, arr in (("state_qpos", state_qpos), ("state_qvel", state_qvel), ("state_mocap_pos", state_mocap_pos), ("state_mocap_quat", state_mocap_quat), ("ball_trajectory", ball_trajectory), ("paddle_trajectory", paddle_trajectory)):
            if not np.all(np.isfinite(arr)):
                failure_reasons.append(f"{name} contains non-finite values")
        if state_qpos.shape[1] != model.nq:
            failure_reasons.append(f"qpos width {state_qpos.shape[1]} != model.nq {model.nq}")
        if state_qvel.shape[1] != model.nv:
            failure_reasons.append(f"qvel width {state_qvel.shape[1]} != model.nv {model.nv}")
        if len({len(state_qpos), len(state_qvel), len(state_mocap_pos), len(state_mocap_quat), len(ball_trajectory), len(paddle_trajectory), len(state_time)}) != 1:
            failure_reasons.append("state and trajectory sample counts differ")
        report["episode_id"] = int(metadata.get("episode_id", int(npz_path.stem.split("_")[-1])))
        report["sample_count"] = n
        report["state_shapes"] = {
            "qpos": list(state_qpos.shape),
            "qvel": list(state_qvel.shape),
            "mocap_pos": list(state_mocap_pos.shape),
            "mocap_quat": list(state_mocap_quat.shape),
        }
        report["state_sequence_importable"] = not failure_reasons
        if n > 0 and report["state_sequence_importable"]:
            if indices and indices > 0:
                pick = np.unique(np.linspace(0, n - 1, min(indices, n)).round().astype(int))
            else:
                pick = np.arange(n, dtype=int)
            max_ball_pos_err = 0.0
            max_paddle_pos_err = 0.0
            max_paddle_quat_err = 0.0
            max_contact_pen = 0.0
            min_contact_dist = float("inf")
            min_ball_center_z = float(np.min(ball_trajectory[:, 3])) if ball_trajectory.size else float("inf")
            for idx in pick:
                mujoco.mj_resetData(model, data)
                data.qpos[:] = state_qpos[idx]
                data.qvel[:] = state_qvel[idx]
                if model.nmocap:
                    data.mocap_pos[:] = state_mocap_pos[idx]
                    data.mocap_quat[:] = state_mocap_quat[idx]
                mujoco.mj_forward(model, data)
                ball_err = float(np.max(np.abs(data.qpos[:3] - ball_trajectory[idx, 1:4])))
                max_ball_pos_err = max(max_ball_pos_err, ball_err)
                if model.nmocap and paddle_trajectory.shape[1] >= 8:
                    max_paddle_pos_err = max(max_paddle_pos_err, float(np.max(np.abs(data.mocap_pos[0] - paddle_trajectory[idx, 1:4]))))
                    max_paddle_quat_err = max(max_paddle_quat_err, _quat_err(data.mocap_quat[0], paddle_trajectory[idx, 4:8]))
                for ci in range(data.ncon):
                    dist = float(data.contact[ci].dist)
                    min_contact_dist = min(min_contact_dist, dist)
                    max_contact_pen = max(max_contact_pen, max(0.0, -dist))
            if not np.isfinite(min_contact_dist):
                min_contact_dist = None
            ground_pen = max(0.0, BALL_RADIUS - min_ball_center_z)
            max_speed = float(np.max(np.linalg.norm(state_qvel[:, :3], axis=1))) if state_qvel.size else 0.0
            max_ang_speed = float(np.max(np.linalg.norm(state_qvel[:, 3:6], axis=1))) if state_qvel.shape[1] >= 6 else 0.0
            report["trajectory_error"] = {
                "max_ball_position_error_m": max_ball_pos_err,
                "max_paddle_position_error_m": max_paddle_pos_err,
                "max_paddle_quaternion_component_error": max_paddle_quat_err,
                "max_linear_speed_m_s": max_speed,
                "max_angular_speed_rad_s": max_ang_speed,
                "thresholds": {
                    "max_ball_position_error_m": MAX_REPLAY_POS_ERR_M,
                    "max_paddle_position_error_m": MAX_PADDLE_POS_ERR_M,
                    "max_linear_speed_m_s": MAX_SPEED_M_S,
                    "max_angular_speed_rad_s": MAX_ANG_SPEED_RAD_S,
                },
            }
            report["contact_error"] = {
                "min_contact_distance_m": min_contact_dist,
                "max_contact_penetration_m": max_contact_pen,
                "threshold_m": MAX_CONTACT_PENETRATION_M,
            }
            report["penetration"] = {
                "ball_ground_penetration_m": ground_pen,
                "min_ball_center_z_m": min_ball_center_z,
                "ball_radius_m": BALL_RADIUS,
                "threshold_m": MAX_GROUND_PENETRATION_M,
            }
            if max_ball_pos_err > MAX_REPLAY_POS_ERR_M:
                failure_reasons.append("recorded ball trajectory does not match imported qpos")
            if max_paddle_pos_err > MAX_PADDLE_POS_ERR_M:
                failure_reasons.append("recorded paddle trajectory does not match imported mocap position")
            if max_contact_pen > MAX_CONTACT_PENETRATION_M:
                failure_reasons.append("contact penetration exceeds threshold")
            if ground_pen > MAX_GROUND_PENETRATION_M:
                failure_reasons.append("ball ground penetration exceeds threshold")
            if max_speed > MAX_SPEED_M_S:
                failure_reasons.append("linear speed exceeds physical sanity threshold")
            if max_ang_speed > MAX_ANG_SPEED_RAD_S:
                failure_reasons.append("angular speed exceeds physical sanity threshold")
        meshes_present = bool(_load_assets(npz_path, metadata))
        descriptions_ok = all(k in z for k in ("scene_description_plain", "scene_description_precise", "object_descriptions"))
        success = bool(metadata.get("success", False))
        paddle_ball_hit = bool(metadata.get("paddle_ball_hit", False))
        wall_hit_world = metadata.get("wall_hit_world_coordinates_m", metadata.get("wall_hit_world"))
        wall_hit_t = metadata.get("wall_hit_time", metadata.get("wall_hit_t"))
        wall_hit_ok = True
        if success:
            wall_hit_ok = wall_hit_world is not None and wall_hit_t is not None and 0.0 <= float(wall_hit_world[2]) <= WALL_HEIGHT + 1e-6
        if not descriptions_ok:
            failure_reasons.append("text descriptions missing")
        if not meshes_present:
            failure_reasons.append("episode mesh assets missing")
        if success and not wall_hit_ok:
            failure_reasons.append("successful wall-hit episode has invalid wall-hit record")
        if metadata.get("sample_mode") == "target_wall" and not paddle_ball_hit:
            failure_reasons.append("target_wall sample did not record paddle-ball hit")
        report["task_constraints"] = {
            "mesh_assets_present": meshes_present,
            "text_descriptions_present": descriptions_ok,
            "success_flag": success,
            "paddle_ball_hit_recorded": paddle_ball_hit,
            "wall_hit_record_valid_when_success": bool(wall_hit_ok),
            "wall_hit_world_coordinates_m": wall_hit_world,
            "wall_hit_time_s": wall_hit_t,
            "sample_mode": metadata.get("sample_mode"),
        }
        report["engine_replay_verified"] = report["state_sequence_importable"] and not any("trajectory" in r or "qpos" in r or "qvel" in r or "mocap" in r for r in failure_reasons)
        report["validation_pass"] = len(failure_reasons) == 0
    except Exception as exc:
        failure_reasons.append(f"exception: {type(exc).__name__}: {exc}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+", help="episode NPZ files to verify")
    ap.add_argument("--indices", type=int, default=0, help="0 checks every stored state sample; positive checks that many evenly spaced samples")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--write-episode-validation", action="store_true")
    ap.add_argument("--allow-fail", action="store_true")
    args = ap.parse_args()
    reports = [verify_episode(Path(p), args.indices) for p in args.npz]
    for r in reports:
        if args.write_episode_validation:
            out_path = _validation_path(Path(r.get("episode_npz", "episode.npz")))
            out_path.write_text(json.dumps(r, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            r["validation_json"] = str(out_path)
    payload = {
        "schema": "state_replay_validation_batch_v1",
        "episodes_checked": len(reports),
        "all_state_reimport_verified": all(r.get("engine_replay_verified") for r in reports),
        "all_validation_passed": all(r.get("validation_pass") for r in reports),
        "reports": reports,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    ok = bool(payload["all_validation_passed"])
    raise SystemExit(0 if (ok or args.allow_fail) else 1)


if __name__ == "__main__":
    main()
