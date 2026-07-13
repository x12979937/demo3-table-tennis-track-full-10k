#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw

import table_tennis_track as base

BALL_MASS = base.BALL_MASS
PADDLE_MASS = 0.18
PADDLE_INERTIA = (0.0022, 0.0012, 0.0025)
WALL_X = base.WALL_X
WALL_HEIGHT = base.WALL_HEIGHT
DT = base.DT
MAX_TIME = base.MAX_TIME
TRAJ_STRIDE = base.TRAJ_STRIDE
PROJECT_ROOT = base.PROJECT_ROOT
EXTERNAL_MODEL_DIR = base.EXTERNAL_MODEL_DIR
DEFAULT_BALL_FRICTION = base.DEFAULT_BALL_FRICTION
DEFAULT_PADDLE_FRICTION = base.DEFAULT_PADDLE_FRICTION
REQUIRED_CAMERAS = ("front", "left", "right", "top", "left_oblique", "right_oblique", "overview")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()




def pack_frame_dir(frame_dir):
    frame_dir = Path(frame_dir)
    tar_path = frame_dir.with_name(frame_dir.name + ".tar.gz")
    tmp_path = frame_dir.with_name(frame_dir.name + ".tar.gz.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    if tar_path.exists():
        tar_path.unlink()
    with tarfile.open(tmp_path, "w:gz") as tf:
        tf.add(frame_dir, arcname=frame_dir.name)
    with tarfile.open(tmp_path, "r:gz") as tf:
        tf.getmembers()
    tmp_path.replace(tar_path)
    shutil.rmtree(frame_dir, ignore_errors=True)
    return tar_path

def load_texture_assets(visual_asset="sketchfab"):
    base.ensure_visual_assets()
    assets = {p.name: p.read_bytes() for p in base.TEXTURE_DIR.glob("*.png")}
    if base.PROCEDURAL_MODEL_DIR.exists():
        for p in base.PROCEDURAL_MODEL_DIR.glob("*.obj"):
            assets[p.name] = p.read_bytes()
    for name in ("sketchfab_paddle_mujoco_visual.obj", "sketchfab_ball_mujoco_visual.obj"):
        mesh_path = EXTERNAL_MODEL_DIR / name
        if not mesh_path.exists():
            raise FileNotFoundError(f"missing professional mesh asset: {mesh_path}")
        assets[name] = mesh_path.read_bytes()
    return assets


def make_xml(case=None, visual_asset="sketchfab"):
    xml = base.make_xml(case, visual_asset="sketchfab")
    xml = xml.replace(
        '<mesh name="sketchfab_paddle_visual_mesh" file="sketchfab_paddle_mujoco_visual.obj"/>',
        '<mesh name="sketchfab_paddle_visual_mesh" file="sketchfab_paddle_mujoco_visual.obj"/>\n'
        '    <mesh name="sketchfab_paddle_collision_mesh" file="sketchfab_paddle_mujoco_visual.obj"/>\n'
        '    <mesh name="sketchfab_ball_collision_mesh" file="sketchfab_ball_mujoco_visual.obj"/>',
    )
    ball_friction = base._friction_text(base._case_friction(case, "ball_friction", DEFAULT_BALL_FRICTION), DEFAULT_BALL_FRICTION)
    paddle_friction = base._friction_text(base._case_friction(case, "paddle_friction", DEFAULT_PADDLE_FRICTION), DEFAULT_PADDLE_FRICTION)
    xml = re.sub(
        r'<geom name="ball_geom" type="sphere"[^\n]*/>',
        f'<geom name="ball_geom" type="mesh" mesh="sketchfab_ball_collision_mesh" mass="{BALL_MASS}" material="ball_mat" condim="4" friction="{ball_friction}" solref="0.0015 0.15" solimp="0.98 0.996 0.0004"/>',
        xml,
    )
    xml = re.sub(
        r'<geom name="paddle_face" type="cylinder"[^\n]*/>',
        f'<geom name="paddle_face" type="mesh" mesh="sketchfab_paddle_collision_mesh" rgba="0.90 0.05 0.05 0.001" condim="4" friction="{paddle_friction}" solref="0.0018 0.22" solimp="0.97 0.996 0.0004"/>',
        xml,
    )
    xml = xml.replace(
        '<body name="paddle" mocap="true" pos="-0.25 0 0.85" quat="1 0 0 0">',
        '<body name="paddle" mocap="true" pos="-0.25 0 0.85" quat="1 0 0 0">\n'
        f'      <inertial pos="0 0 -0.06" mass="{PADDLE_MASS}" diaginertia="{PADDLE_INERTIA[0]} {PADDLE_INERTIA[1]} {PADDLE_INERTIA[2]}"/>',
    )
    return xml


def sample_case(rng, case_id, mode="balanced"):
    case = base.sample_case(rng, case_id, mode)
    p0 = np.array(case["paddle_start"], dtype=float)
    ph = np.array(case["paddle_hit_pose"], dtype=float)
    p0[1] += float(rng.uniform(-0.36, 0.36))
    p0[0] += float(rng.uniform(-0.06, 0.08))
    p0[2] = float(np.clip(p0[2] + rng.uniform(-0.035, 0.035), 0.58, 1.38))
    case["paddle_start"] = p0.astype(float).tolist()
    duration = max(1e-4, float(case["planned_t_hit"] - case.get("move_t_start", 0.0)))
    vel = (ph - p0) / duration
    speed = float(np.linalg.norm(vel))
    case["paddle_velocity_xyz"] = vel.astype(float).tolist()
    case["paddle_speed"] = speed
    case["paddle_velocity_dir"] = (vel / max(1e-9, speed)).astype(float).tolist()
    ball_xy = np.array(case["ball_initial_pos"][:2], dtype=float)
    rel = ball_xy - p0[:2]
    case["paddle_start_position"] = case["paddle_start"]
    case["paddle_target_end_position"] = case["paddle_hit_pose"]
    case["ball_paddle_start_distance_m"] = float(np.linalg.norm([rel[0], rel[1], case["ball_initial_pos"][2] - p0[2]]))
    case["ball_paddle_ground_distance_m"] = float(np.linalg.norm(rel))
    case["ball_paddle_ground_angle_rad"] = float(math.atan2(rel[1], rel[0]))
    case["ball_paddle_ground_angle_deg"] = float(math.degrees(case["ball_paddle_ground_angle_rad"]))
    case["ball_paddle_friction_sliding"] = float(case.get("paddle_friction", DEFAULT_PADDLE_FRICTION)[0])
    case["paddle_euler_rad"] = [float(case["paddle_roll"]), float(case["paddle_pitch"]), float(case["paddle_yaw"])]
    case["paddle_euler_deg"] = [float(math.degrees(v)) for v in case["paddle_euler_rad"]]
    return case


def object_and_scene_descriptions(case):
    plain_scene = (
        "An indoor MuJoCo table-tennis impact scene: a professional ping-pong paddle starts from a sampled "
        "3D position and Euler-angle pose, moves along an oblique path, strikes a falling ping-pong ball, "
        "and the ball may hit a finite target wall. Wooden floor and indoor wall textures are used."
    )
    precise_scene = {
        "task": "Demo3.TableTennisTrack",
        "engine": "MuJoCo",
        "coordinate_frame": {"x": "paddle-to-wall", "y": "right", "z": "up", "wall_plane": f"x={WALL_X}"},
        "randomized_parameters": {
            "paddle_euler_rad": case.get("paddle_euler_rad"),
            "paddle_start_position_m": case.get("paddle_start"),
            "paddle_velocity_m_s": case.get("paddle_velocity_xyz"),
            "ball_initial_height_m": case.get("drop_height_m"),
            "ball_paddle_start_distance_m": case.get("ball_paddle_start_distance_m"),
            "ground_projection_angle_deg": case.get("ball_paddle_ground_angle_deg"),
            "paddle_ball_friction_sliding": case.get("ball_paddle_friction_sliding"),
        },
        "events": ["ball_drop", "paddle_motion", "paddle_ball_contact_or_miss", "wall_contact_or_miss"],
    }
    objects = {
        "paddle": {
            "plain": "A professional table-tennis paddle with downloaded mesh geometry, red rubber face and wooden handle; it is a kinematic striking tool with recorded mass, pose, velocity and momentum.",
            "precise": {"type": "rigid_kinematic_tool", "mesh_visual": "sketchfab_paddle_mujoco_visual.obj", "mesh_collision": "sketchfab_paddle_mujoco_visual.obj", "mass_kg": PADDLE_MASS, "inertia_diag_kg_m2": list(PADDLE_INERTIA)},
        },
        "ball": {
            "plain": "A ping-pong ball using the downloaded spherical mesh as both visual and collision geometry; its full trajectory, velocity, angular velocity, spin and momentum are recorded.",
            "precise": {"type": "free_rigid_body", "mesh_visual": "sketchfab_ball_mujoco_visual.obj", "mesh_collision": "sketchfab_ball_mujoco_visual.obj", "mass_kg": BALL_MASS, "radius_m": base.BALL_RADIUS},
        },
        "target_wall": {
            "plain": "A finite textured wall target that receives or misses the ball impact.",
            "precise": {"type": "static_box", "plane_x_m": WALL_X, "height_m": WALL_HEIGHT, "width_m": base.WALL_WIDTH},
        },
    }
    return {"scene_plain_language": plain_scene, "scene_precise_language": precise_scene, "objects": objects}


def mask_to_bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def mask_to_rle(mask):
    flat = np.asarray(mask, dtype=np.uint8).ravel(order="F")
    counts = []
    last = 0
    run = 0
    for v in flat:
        v = int(v)
        if v == last:
            run += 1
        else:
            counts.append(run)
            run = 1
            last = v
    counts.append(run)
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": counts, "order": "F", "format": "uncompressed_coco_rle"}


def object_geom_ids(model):
    ids = {"ball": [], "paddle": []}
    for gid in range(model.ngeom):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])) or ""
        if bname in ids:
            ids[bname].append(gid)
    return ids


def annotate_rgb(rgb, masks, bboxes):
    out = rgb.copy()
    colors = {"ball": np.array([60, 170, 255], dtype=np.float32), "paddle": np.array([255, 70, 70], dtype=np.float32)}
    for name, mask in masks.items():
        if mask is None or not mask.any():
            continue
        out[mask] = np.clip(out[mask].astype(np.float32) * 0.55 + colors[name] * 0.45, 0, 255).astype(np.uint8)
    im = Image.fromarray(out)
    draw = ImageDraw.Draw(im)
    labels = {"ball": "pingpong_ball", "paddle": "paddle"}
    for name, bbox in bboxes.items():
        if not bbox:
            continue
        color = (60, 170, 255) if name == "ball" else (255, 70, 70)
        draw.rectangle(bbox, outline=color, width=3)
        draw.text((bbox[0] + 2, max(0, bbox[1] - 16)), labels[name], fill=color)
    return np.asarray(im)


def render_rgb_and_masks(renderer, model, data, camera, geom_ids):
    renderer.disable_segmentation_rendering()
    renderer.update_scene(data, camera=camera)
    rgb = renderer.render().copy()
    renderer.enable_segmentation_rendering()
    renderer.update_scene(data, camera=camera)
    seg = renderer.render().copy()
    renderer.disable_segmentation_rendering()
    geom = seg[:, :, 0]
    masks = {name: np.isin(geom, ids) for name, ids in geom_ids.items()}
    bboxes = {name: mask_to_bbox(mask) for name, mask in masks.items()}
    ann = {
        "camera": camera,
        "objects": {
            name: {"bbox_xyxy": bboxes[name], "mask_area_px": int(masks[name].sum()), "mask_rle": mask_to_rle(masks[name])}
            for name in masks
        },
    }
    return rgb, masks, bboxes, ann


def copy_episode_meshes(out, eid):
    ep_dir = Path(out) / f"episode_{eid:06d}_meshes"
    ep_dir.mkdir(parents=True, exist_ok=True)
    assets = {
        "sketchfab_paddle_mujoco_visual.obj": EXTERNAL_MODEL_DIR / "sketchfab_paddle_mujoco_visual.obj",
        "sketchfab_ball_mujoco_visual.obj": EXTERNAL_MODEL_DIR / "sketchfab_ball_mujoco_visual.obj",
    }
    role_map = {
        "paddle_visual": "sketchfab_paddle_mujoco_visual.obj",
        "paddle_collision": "sketchfab_paddle_mujoco_visual.obj",
        "ball_visual": "sketchfab_ball_mujoco_visual.obj",
        "ball_collision": "sketchfab_ball_mujoco_visual.obj",
    }
    copied = {}
    for filename, src in assets.items():
        dst = ep_dir / filename
        shutil.copy2(src, dst)
        copied[filename] = {"file": filename, "sha256": sha256_file(dst), "source": str(src)}
    manifest = [{"role": role, **copied[filename]} for role, filename in role_map.items()]
    (ep_dir / "mesh_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return ep_dir, manifest


def physical_properties(model, case):
    ball_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    paddle_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "paddle")
    return {
        "ball": {
            "mass_kg": float(model.body_mass[ball_bid]) if ball_bid >= 0 else BALL_MASS,
            "inertia_diag_kg_m2": model.body_inertia[ball_bid].astype(float).tolist() if ball_bid >= 0 else None,
            "friction": case.get("ball_friction", DEFAULT_BALL_FRICTION),
            "mesh_visual": "sketchfab_ball_mujoco_visual.obj",
            "mesh_collision": "sketchfab_ball_mujoco_visual.obj",
        },
        "paddle": {
            "mass_kg": float(model.body_mass[paddle_bid]) if paddle_bid >= 0 else PADDLE_MASS,
            "configured_mass_kg": PADDLE_MASS,
            "inertia_diag_kg_m2": model.body_inertia[paddle_bid].astype(float).tolist() if paddle_bid >= 0 else list(PADDLE_INERTIA),
            "friction": case.get("paddle_friction", DEFAULT_PADDLE_FRICTION),
            "mesh_visual": "sketchfab_paddle_mujoco_visual.obj",
            "mesh_collision": "sketchfab_paddle_mujoco_visual.obj",
        },
        "wall": {"x_m": WALL_X, "height_m": WALL_HEIGHT, "width_m": base.WALL_WIDTH},
    }


def run_episode(case, out_dir=None, render=False, camera="overview", width=1280, height=720, video_tag=None, save_npz=True, visual_asset="sketchfab", copy_meshes=False, save_raw_rgb_frames=False):
    model = mujoco.MjModel.from_xml_string(make_xml(case, visual_asset=visual_asset), assets=load_texture_assets(visual_asset))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    base.set_ball_state(model, data, case["ball_initial_pos"], case["ball_initial_vel"], case.get("ball_initial_angular_vel", [0, 0, 0]))
    p, q = base.paddle_pose(case, 0.0)
    data.mocap_pos[0] = p
    data.mocap_quat[0] = q
    mujoco.mj_forward(model, data)
    wall_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wall")
    ball_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    paddle_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "paddle_face")
    renderer = None
    frames = []
    perception = []
    render_every = 8
    geom_ids = object_geom_ids(model)
    raw_frame_dir = None
    raw_frame_archive = None
    if render and save_raw_rgb_frames and out_dir is not None:
        tag_for_frames = video_tag or camera
        raw_frame_dir = Path(out_dir) / f"episode_{int(case['episode_id']):06d}_{tag_for_frames}_raw_rgb_frames"
        if raw_frame_dir.exists():
            shutil.rmtree(raw_frame_dir, ignore_errors=True)
        raw_frame_dir.mkdir(parents=True, exist_ok=True)
    if render:
        renderer = mujoco.Renderer(model, height=height, width=width)
    rows, ball_rows, paddle_rows = [], [], []
    state_times, state_qpos, state_qvel, state_mocap_pos, state_mocap_quat = [], [], [], [], []
    contacts = []
    hit = None
    first_paddle_contact = None
    miss_reason = None
    min_ball_wall_abs = 999.0
    final_p = p.copy()
    final_q = q.copy()

    def scan_ball_contacts(t, paddle_pos):
        nonlocal hit, first_paddle_contact
        for ci in range(data.ncon):
            c = data.contact[ci]
            gs = {int(c.geom1), int(c.geom2)}
            if ball_gid not in gs:
                continue
            other_gid = int(c.geom2) if int(c.geom1) == ball_gid else int(c.geom1)
            other_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other_gid) or f"geom_{other_gid}"
            if other_gid == paddle_gid or other_name.startswith("paddle"):
                ev = {"t": float(t), "type": "ball_paddle", "geom": other_name, "pos": np.array(c.pos).astype(float).tolist()}
                contacts.append(ev)
                if first_paddle_contact is None:
                    first_paddle_contact = ev
            if other_gid == wall_gid and hit is None:
                hp = np.array(c.pos, dtype=float)
                hit = {
                    "t": float(t),
                    "world": hp.tolist(),
                    "local": [float(hp[1] - paddle_pos[1]), float(hp[2] - paddle_pos[2])],
                    "wall_uv_m": [float(hp[1]), float(hp[2])],
                    "wall_centered_m": [float(hp[1]), float(hp[2] - WALL_HEIGHT / 2.0)],
                }

    render_tail_s = 0.85 if render else 0.18
    stop_after_t = None
    for i in range(int(MAX_TIME / DT) + 1):
        t = i * DT
        p, q = base.paddle_pose(case, t)
        final_p = p.copy()
        final_q = q.copy()
        data.mocap_pos[0] = p
        data.mocap_quat[0] = q
        mujoco.mj_forward(model, data)
        scan_ball_contacts(t, p)
        mujoco.mj_step(model, data)
        scan_ball_contacts(t, p)
        bpos, bvel, bangvel = base.get_ball_state(model, data)
        pvel = np.array(case.get("paddle_velocity_xyz", [0, 0, 0]), dtype=float) if t <= case.get("planned_t_hit", 0.0) else np.zeros(3)
        min_ball_wall_abs = min(min_ball_wall_abs, abs(float(WALL_X - bpos[0])))
        if i % TRAJ_STRIDE == 0:
            rows.append([t, *bpos.tolist(), *bvel.tolist(), *bangvel.tolist(), *p.tolist(), *q.tolist()])
            ball_rows.append([t, *bpos.tolist(), *bvel.tolist(), *bangvel.tolist(), *(BALL_MASS * bvel).tolist()])
            paddle_rows.append([t, *p.tolist(), *q.tolist(), *pvel.tolist(), *(PADDLE_MASS * pvel).tolist()])
            state_times.append(t)
            state_qpos.append(data.qpos.copy())
            state_qvel.append(data.qvel.copy())
            state_mocap_pos.append(data.mocap_pos.copy())
            state_mocap_quat.append(data.mocap_quat.copy())
        if render and i % render_every == 0:
            rgb, masks, bboxes, ann = render_rgb_and_masks(renderer, model, data, camera, geom_ids)
            frame_index = len(frames)
            if raw_frame_dir is not None:
                Image.fromarray(rgb).save(raw_frame_dir / f"frame_{frame_index:05d}.png")
            ann["t"] = float(t)
            ann["frame_index"] = frame_index
            ann["image_width"] = int(width)
            ann["image_height"] = int(height)
            perception.append(ann)
            labeled = annotate_rgb(rgb, masks, bboxes)
            frames.append(base.overlay_frame(labeled, {**case, "success": bool(hit), "miss_reason": miss_reason, "wall_hit_local": hit["local"] if hit else None, "wall_hit_world": hit["world"] if hit else None, "sample_mode": case.get("sample_mode")}, t, camera, len(rows)))
        if hit is not None and stop_after_t is None:
            stop_after_t = hit["t"] + render_tail_s
        if stop_after_t is not None and t > stop_after_t:
            break
        if hit is None and miss_reason is None:
            if bpos[2] < base.BALL_RADIUS * 0.55 and t > 0.05:
                miss_reason = "ground_before_wall"
                if render:
                    stop_after_t = t + render_tail_s
                else:
                    break
            elif bpos[0] > WALL_X + 0.08:
                miss_reason = "over_wall" if bpos[2] > WALL_HEIGHT else "passed_wall_without_contact"
                if render:
                    stop_after_t = t + render_tail_s
                else:
                    break
            elif t >= MAX_TIME:
                miss_reason = "timeout"
                break
    traj = np.array(rows, dtype=np.float64)
    ball_traj = np.array(ball_rows, dtype=np.float64)
    paddle_traj = np.array(paddle_rows, dtype=np.float64)
    success = hit is not None and 0.0 <= hit["world"][2] <= WALL_HEIGHT
    if not success and miss_reason is None:
        miss_reason = "wall_contact_outside_height" if hit is not None else "unknown_miss"
    descriptions = object_and_scene_descriptions(case)
    final_ball_position = bpos.astype(float).tolist()
    final_ball_velocity = bvel.astype(float).tolist()
    final_ball_angular_velocity = bangvel.astype(float).tolist()
    paddle_initial_velocity = np.array(case.get("paddle_velocity_xyz", [0, 0, 0]), dtype=float)
    result = {
        **case,
        "success": bool(success),
        "paddle_ball_hit": first_paddle_contact is not None,
        "ball_start_position": [float(v) for v in case["ball_initial_pos"]],
        "ball_start_height": float(case["ball_initial_pos"][2]),
        "ball_start_velocity": [float(v) for v in case["ball_initial_vel"]],
        "ball_end_position": final_ball_position,
        "ball_end_velocity": final_ball_velocity,
        "ball_end_angular_velocity": final_ball_angular_velocity,
        "ball_mass": float(BALL_MASS),
        "ball_initial_momentum_kg_m_s": (BALL_MASS * np.array(case["ball_initial_vel"], dtype=float)).astype(float).tolist(),
        "ball_final_momentum_kg_m_s": (BALL_MASS * bvel).astype(float).tolist(),
        "paddle_start_position": [float(v) for v in case["paddle_start"]],
        "paddle_end_position": final_p.astype(float).tolist(),
        "paddle_start_quat": base.rpy_quat(case["paddle_roll"], case["paddle_pitch"], case["paddle_yaw"]).astype(float).tolist(),
        "paddle_end_quat": final_q.astype(float).tolist(),
        "paddle_start_euler_rad": case.get("paddle_euler_rad"),
        "paddle_start_euler_deg": case.get("paddle_euler_deg"),
        "paddle_end_euler_rad": case.get("paddle_euler_rad"),
        "paddle_end_euler_deg": case.get("paddle_euler_deg"),
        "paddle_mass": float(PADDLE_MASS),
        "paddle_initial_momentum_kg_m_s": (PADDLE_MASS * paddle_initial_velocity).astype(float).tolist(),
        "paddle_final_momentum_kg_m_s": [0.0, 0.0, 0.0],
        "wall_hit_world": hit["world"] if hit else None,
        "wall_hit_world_coordinates_m": hit["world"] if hit else None,
        "wall_hit_local": hit["local"] if hit else None,
        "wall_hit_wall_uv_m": hit["wall_uv_m"] if hit else None,
        "wall_hit_wall_coordinates_m": hit["wall_uv_m"] if hit else None,
        "wall_hit_wall_centered_m": hit["wall_centered_m"] if hit else None,
        "wall_hit_t": hit["t"] if hit else None,
        "wall_hit_time": hit["t"] if hit else None,
        "miss_reason": None if success else miss_reason,
        "first_paddle_contact": first_paddle_contact,
        "contact_events": contacts[:96],
        "min_ball_wall_abs": float(min_ball_wall_abs),
        "object_physical_properties": physical_properties(model, case),
        "mesh_policy": "visual and collision geoms use downloaded professional mesh OBJ assets for ball and paddle",
        "state_replay": {"format": "MuJoCo qpos/qvel plus mocap state sampled at trajectory_hz", "qpos_size": int(model.nq), "qvel_size": int(model.nv), "mocap_count": int(model.nmocap)},
        "descriptions": descriptions,
        "wall_x": WALL_X,
        "wall_height": WALL_HEIGHT,
        "dt": DT,
        "trajectory_hz": int(1 / (DT * TRAJ_STRIDE)),
        "trajectory_columns": ["t", "ball_x", "ball_y", "ball_z", "ball_vx", "ball_vy", "ball_vz", "ball_wx", "ball_wy", "ball_wz", "paddle_x", "paddle_y", "paddle_z", "paddle_qw", "paddle_qx", "paddle_qy", "paddle_qz"],
        "ball_trajectory_columns": ["t", "x", "y", "z", "vx", "vy", "vz", "wx", "wy", "wz", "px", "py", "pz"],
        "paddle_trajectory_columns": ["t", "x", "y", "z", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "px", "py", "pz"],
    }
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        eid_num = int(case["episode_id"])
        eid = f"episode_{eid_num:06d}"
        mesh_manifest = []
        if copy_meshes and save_npz:
            mesh_dir, mesh_manifest = copy_episode_meshes(out, eid_num)
            result["episode_mesh_dir"] = str(mesh_dir.resolve())
            result["episode_mesh_manifest"] = mesh_manifest
        if save_npz:
            (out / f"{eid}_descriptions.json").write_text(json.dumps(descriptions, indent=2, ensure_ascii=False), encoding="utf-8")
            np.savez_compressed(
                out / f"{eid}.npz",
                trajectory=traj,
                ball_trajectory=ball_traj,
                paddle_trajectory=paddle_traj,
                state_time=np.array(state_times, dtype=np.float64),
                state_qpos=np.array(state_qpos, dtype=np.float64),
                state_qvel=np.array(state_qvel, dtype=np.float64),
                state_mocap_pos=np.array(state_mocap_pos, dtype=np.float64),
                state_mocap_quat=np.array(state_mocap_quat, dtype=np.float64),
                metadata=json.dumps(result, ensure_ascii=False),
                scene_description_plain=np.array([descriptions["scene_plain_language"]]),
                scene_description_precise=json.dumps(descriptions["scene_precise_language"], ensure_ascii=False),
                object_descriptions=json.dumps(descriptions["objects"], ensure_ascii=False),
                model_xml=make_xml(case, visual_asset=visual_asset),
                visual_asset=np.array([visual_asset]),
                mesh_manifest=json.dumps(mesh_manifest, ensure_ascii=False),
                paddle_control=np.array([
                    case["paddle_speed"], *np.array(case.get("paddle_velocity_xyz", [0, 0, 0]), dtype=float).tolist(),
                    case["paddle_roll"], case["paddle_pitch"], case["paddle_yaw"],
                    case["move_t_start"], case["planned_t_hit"], case["move_duration"],
                    *case.get("ball_friction", DEFAULT_BALL_FRICTION), *case.get("paddle_friction", DEFAULT_PADDLE_FRICTION),
                    *case.get("ball_spin_axis", [0, 0, 1]), case.get("ball_spin_speed_rad_s", 0.0),
                    *case.get("ball_initial_angular_vel", [0, 0, 0]), *case["ball_initial_pos"],
                    case.get("ball_paddle_ground_angle_rad", 0.0), case.get("ball_paddle_start_distance_m", 0.0), case.get("ball_paddle_friction_sliding", 0.0),
                ], dtype=np.float64),
                paddle_control_columns=np.array([
                    "paddle_speed_m_s", "paddle_vx_m_s", "paddle_vy_m_s", "paddle_vz_m_s",
                    "paddle_roll_rad", "paddle_pitch_rad", "paddle_yaw_rad",
                    "paddle_move_start_time_s", "paddle_planned_hit_time_s", "paddle_move_duration_s",
                    "ball_friction_sliding", "ball_friction_torsional", "ball_friction_rolling",
                    "paddle_friction_sliding", "paddle_friction_torsional", "paddle_friction_rolling",
                    "ball_spin_axis_x", "ball_spin_axis_y", "ball_spin_axis_z", "ball_spin_speed_rad_s",
                    "ball_initial_wx_rad_s", "ball_initial_wy_rad_s", "ball_initial_wz_rad_s",
                    "ball_initial_x_m", "ball_initial_y_m", "ball_initial_z_m",
                    "ball_paddle_ground_angle_rad", "ball_paddle_start_distance_m", "ball_paddle_friction_sliding",
                ]),
                paddle_start_pose=np.array([*case["paddle_start"], *base.rpy_quat(case["paddle_roll"], case["paddle_pitch"], case["paddle_yaw"])], dtype=np.float64),
                paddle_end_pose=np.array([*final_p.tolist(), *final_q.tolist()], dtype=np.float64),
                paddle_hit_pose=np.array([*case["paddle_hit_pose"], *base.rpy_quat(case["paddle_roll"], case["paddle_pitch"], case["paddle_yaw"])], dtype=np.float64),
                paddle_start_euler_rad=np.array(case.get("paddle_euler_rad"), dtype=np.float64),
                paddle_end_euler_rad=np.array(case.get("paddle_euler_rad"), dtype=np.float64),
                paddle_velocity_dir=np.array(case["paddle_velocity_dir"], dtype=np.float64),
                paddle_velocity_xyz=np.array(case["paddle_velocity_xyz"], dtype=np.float64),
                ball_initial_pos=np.array(case["ball_initial_pos"], dtype=np.float64),
                ball_initial_vel=np.array(case["ball_initial_vel"], dtype=np.float64),
                ball_initial_angular_vel=np.array(case.get("ball_initial_angular_vel", [0, 0, 0]), dtype=np.float64),
                ball_spin_axis=np.array(case.get("ball_spin_axis", [0, 0, 1]), dtype=np.float64),
                ball_spin_speed_rad_s=np.array([case.get("ball_spin_speed_rad_s", 0.0)], dtype=np.float64),
                ball_friction=np.array(case.get("ball_friction", DEFAULT_BALL_FRICTION), dtype=np.float64),
                paddle_friction=np.array(case.get("paddle_friction", DEFAULT_PADDLE_FRICTION), dtype=np.float64),
                ball_mass_kg=np.array([BALL_MASS], dtype=np.float64),
                paddle_mass_kg=np.array([PADDLE_MASS], dtype=np.float64),
            )
        if render and frames:
            tagged = video_tag or camera
            video_path = out / f"{eid}_{tagged}.mp4"
            imageio.mimsave(video_path, frames, fps=int(1 / (DT * render_every)), quality=9, macro_block_size=16)
            if raw_frame_dir is not None and raw_frame_dir.exists():
                raw_frame_archive = pack_frame_dir(raw_frame_dir)
            perception_doc = {"episode_id": eid_num, "camera": camera, "video_file": str(video_path.name), "raw_rgb_frame_archive": str(raw_frame_archive.name) if raw_frame_archive else None, "objects": ["pingpong_ball", "paddle"], "mask_format": "uncompressed_coco_rle", "frames": perception}
            with gzip.open(out / f"{eid}_{tagged}_perception.json.gz", "wt", encoding="utf-8") as f:
                json.dump(perception_doc, f, ensure_ascii=False)
    if renderer is not None:
        renderer.close()
    return result


def index_record(result, out, cameras):
    eid = int(result["episode_id"])
    ep = f"episode_{eid:06d}"
    return {
        "episode_id": eid,
        "episode_npz": str((Path(out) / f"{ep}.npz").resolve()),
        "sample_mode": result.get("sample_mode"),
        "success": bool(result.get("success")),
        "paddle_ball_hit": bool(result.get("paddle_ball_hit")),
        "miss_reason": result.get("miss_reason"),
        "wall_hit_world": result.get("wall_hit_world"),
        "wall_hit_wall_uv_m": result.get("wall_hit_wall_uv_m"),
        "wall_hit_wall_centered_m": result.get("wall_hit_wall_centered_m"),
        "wall_hit_t": result.get("wall_hit_t"),
        "ball_initial_pos": result.get("ball_initial_pos"),
        "ball_initial_vel": result.get("ball_initial_vel"),
        "ball_initial_angular_vel": result.get("ball_initial_angular_vel"),
        "ball_spin_axis": result.get("ball_spin_axis"),
        "ball_spin_speed_rad_s": result.get("ball_spin_speed_rad_s"),
        "ball_friction": result.get("ball_friction"),
        "paddle_friction": result.get("paddle_friction"),
        "ball_paddle_ground_angle_deg": result.get("ball_paddle_ground_angle_deg"),
        "ball_paddle_start_distance_m": result.get("ball_paddle_start_distance_m"),
        "paddle_start": result.get("paddle_start"),
        "paddle_end_position": result.get("paddle_end_position"),
        "paddle_velocity_xyz": result.get("paddle_velocity_xyz"),
        "paddle_euler_rad": result.get("paddle_euler_rad"),
        "trajectory_columns": result.get("trajectory_columns"),
        "state_replay": result.get("state_replay"),
        "mesh_dir": result.get("episode_mesh_dir"),
        "description_json": str((Path(out) / f"{ep}_descriptions.json").resolve()),
        "videos": {cam: str((Path(out) / f"{ep}_{cam}.mp4").resolve()) for cam in cameras},
        "perception_annotations": {cam: str((Path(out) / f"{ep}_{cam}_perception.json.gz").resolve()) for cam in cameras},
        "raw_rgb_frame_archives": {cam: str((Path(out) / f"{ep}_{cam}_raw_rgb_frames.tar.gz").resolve()) for cam in cameras},
        "state_replay_validation_json": str((Path(out) / f"{ep}_state_replay_validation.json").resolve()),
    }


def cleanup_episode_outputs(out, episode_id):
    out = Path(out)
    ep = f"episode_{int(episode_id):06d}"
    for path in out.glob(f"{ep}*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            try:
                path.unlink()
            except FileNotFoundError:
                pass


def validate_episode_state_replay(npz_path):
    validator = Path(__file__).resolve().with_name("verify_state_replay.py")
    cmd = [sys.executable, str(validator), str(npz_path), "--json", "--write-episode-validation"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    log_path = Path(npz_path).with_name(f"{Path(npz_path).stem}_state_replay_validation.log")
    log_path.write_text(proc.stdout, encoding="utf-8")
    return proc.returncode == 0


def write_manifest_and_index(out, results, cameras):
    out = Path(out)
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as f_manifest, (out / "dataset_index.jsonl").open("w", encoding="utf-8") as f_index:
        for r in sorted(results, key=lambda x: int(x.get("episode_id", 0))):
            f_manifest.write(json.dumps(r, ensure_ascii=False) + "\n")
            f_index.write(json.dumps(index_record(r, out, cameras), ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=str, default="Outputs/full_dataset")
    ap.add_argument("--videos", type=int, default=6)
    ap.add_argument("--video-cameras", type=str, default=",".join(REQUIRED_CAMERAS))
    ap.add_argument("--video-stride", type=int, default=0)
    ap.add_argument("--start-id", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--sample-ratios", type=str, default="target_wall=4,ground_miss=2,over_wall=2,random=1")
    ap.add_argument("--progress-every", type=int, default=50)
    ap.add_argument("--mode", choices=["balanced", "target_wall", "ground_miss", "over_wall", "random"], default="balanced")
    ap.add_argument("--showcase", action="store_true")
    ap.add_argument("--visual-asset", choices=["sketchfab"], default="sketchfab")
    ap.add_argument("--copy-episode-meshes", action="store_true")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--keep-raw-rgb-frames", action="store_true", help="save raw renderer RGB PNG frames for each rendered camera and pack them into per-camera tar.gz archives")
    ap.add_argument("--no-state-replay-validate", dest="state_replay_validate", action="store_false", help="disable per-episode MuJoCo state replay validation")
    ap.add_argument("--validation-retries", type=int, default=2, help="regenerate an episode this many times if replay validation fails")
    ap.set_defaults(state_replay_validate=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cameras = [c.strip() for c in args.video_cameras.split(",") if c.strip()]
    ratios = base.parse_sample_ratios(args.sample_ratios)
    cases = []
    for ep in range(args.start_id, args.start_id + args.episodes):
        case_mode = base.draw_mode(rng, ep, args.mode, ratios)
        cases.append(sample_case(rng, ep, case_mode))
    summary = []
    for idx, case in enumerate(cases, start=1):
        attempts = 0
        while True:
            npz_path = out / f"episode_{case['episode_id']:06d}.npz"
            if args.resume and npz_path.exists():
                res = base.load_existing_result(npz_path)
            else:
                res = run_episode(case, out, render=False, save_npz=True, visual_asset=args.visual_asset, copy_meshes=args.copy_episode_meshes)
            if not args.state_replay_validate or validate_episode_state_replay(npz_path):
                break
            cleanup_episode_outputs(out, case["episode_id"])
            attempts += 1
            if attempts > args.validation_retries:
                raise RuntimeError(f"state replay validation failed for episode {case['episode_id']} after {attempts} attempts")
            case = sample_case(rng, int(case["episode_id"]), case.get("sample_mode", args.mode if args.mode != "balanced" else base.draw_mode(rng, int(case["episode_id"]), args.mode, ratios)))
        summary.append(res)
        if args.progress_every > 0 and (idx % args.progress_every == 0 or idx == len(cases)):
            succ = sum(1 for r in summary if r.get("success"))
            print(f"[progress] {idx}/{len(cases)} episodes, success_rate={succ/max(1,idx):.3f}, out={out}", flush=True)
    if args.video_stride > 0:
        video_cases = [c for idx, c in enumerate(cases) if idx % args.video_stride == 0][:max(0, args.videos)]
    else:
        video_cases = cases[:max(0, args.videos)]
    for v_idx, case in enumerate(video_cases, start=1):
        for cam in cameras:
            run_episode(case, out, render=True, camera=cam, width=args.width, height=args.height, video_tag=cam, save_npz=False, visual_asset=args.visual_asset, save_raw_rgb_frames=args.keep_raw_rgb_frames)
        if args.progress_every > 0 and (v_idx % max(1, min(args.progress_every, 10)) == 0 or v_idx == len(video_cases)):
            print(f"[video] {v_idx}/{len(video_cases)} episodes, cameras={len(cameras)}, out={out}", flush=True)
    if args.showcase:
        picked = []
        seen = set()
        for case in cases:
            if case["sample_mode"] not in seen:
                picked.append(case)
                seen.add(case["sample_mode"])
            if len(picked) >= 3:
                break
        for case in picked:
            c2 = dict(case)
            c2["episode_id"] = 100000 + int(case["episode_id"])
            for cam in cameras:
                run_episode(c2, out, render=True, camera=cam, width=args.width, height=args.height, video_tag=f"showcase_{case['sample_mode']}_{cam}", save_npz=False, visual_asset=args.visual_asset, save_raw_rgb_frames=args.keep_raw_rgb_frames)
    write_manifest_and_index(out, summary, cameras)
    base.write_summary(out, summary)


if __name__ == "__main__":
    main()
