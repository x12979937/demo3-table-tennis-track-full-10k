#!/usr/bin/env python3
import argparse, json, math, os, time
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import mujoco
import imageio.v2 as imageio
from PIL import Image, ImageDraw

G = 9.81
BALL_RADIUS = 0.020
BALL_MASS = 0.0027
PADDLE_RADIUS = 0.115
PADDLE_THICKNESS = 0.008
WALL_X = 2.0
WALL_HEIGHT = 1.35
WALL_WIDTH = 20.0
DT = 0.001
MAX_TIME = 3.0
TRAJ_STRIDE = 5
PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXTURE_DIR = PROJECT_ROOT / "assets" / "textures"
PROCEDURAL_MODEL_DIR = PROJECT_ROOT / "assets" / "procedural_models"
EXTERNAL_MODEL_DIR = PROJECT_ROOT / "assets" / "external_models" / "sketchfab_andrewhunt95_table_tennis_rackets_ping_pong" / "converted"
DEFAULT_BALL_FRICTION = (0.06, 0.004, 0.0001)
DEFAULT_PADDLE_FRICTION = (1.05, 0.02, 0.001)


def _save_texture(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def ensure_visual_assets():
    TEXTURE_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)

    # Warm indoor wooden floor with plank seams and mild grain.
    img = Image.new("RGB", (1024, 1024), (153, 104, 58)); draw = ImageDraw.Draw(img)
    for y in range(0, 1024, 128):
        base = int(135 + rng.integers(-12, 18)); color = (base, int(base*0.68), int(base*0.36))
        draw.rectangle([0, y, 1024, y+127], fill=color)
        draw.line([0, y, 1024, y], fill=(78, 48, 26), width=3)
        offset = 0 if (y//128) % 2 == 0 else 180
        for x in range(-offset, 1024, 260):
            draw.line([x, y, x, y+127], fill=(90, 56, 31), width=2)
    for _ in range(900):
        x=int(rng.integers(0,1024)); y=int(rng.integers(0,1024)); l=int(rng.integers(18,90))
        c=int(rng.integers(85,145)); draw.line([x,y,min(1023,x+l),y+int(rng.integers(-2,3))], fill=(c, int(c*0.62), int(c*0.34)), width=1)
    _save_texture(TEXTURE_DIR/"wood_floor.png", img)

    # Soft indoor wall plaster.
    img = Image.new("RGB", (768, 768), (206, 209, 206)); draw = ImageDraw.Draw(img)
    for _ in range(2500):
        x=int(rng.integers(0,768)); y=int(rng.integers(0,768)); v=int(rng.integers(-10,11));
        col=(max(0,min(255,206+v)), max(0,min(255,209+v)), max(0,min(255,206+v)))
        draw.point((x,y), fill=col)
    _save_texture(TEXTURE_DIR/"indoor_wall.png", img)

    # Target wall: painted indoor target panel with visible scoring grid.
    img = Image.new("RGB", (1024, 768), (168, 184, 205)); draw = ImageDraw.Draw(img)
    for y in range(0, 768, 24):
        shade = 166 + int(10 * math.sin(y / 47.0))
        draw.line([0, y, 1024, y], fill=(shade, min(220, shade+15), min(235, shade+30)), width=2)
    for x in range(0,1024,64): draw.line([x,0,x,768], fill=(94,118,150), width=2)
    for y in range(0,768,64): draw.line([0,y,1024,y], fill=(94,118,150), width=2)
    draw.rectangle([0, 0, 1024, 36], fill=(28, 70, 125))
    draw.rectangle([38, 54, 986, 728], outline=(28, 70, 125), width=10)
    draw.ellipse([392, 250, 632, 490], outline=(196, 66, 54), width=12)
    draw.line([512,80,512,705], fill=(45,80,130), width=5)
    draw.line([80,384,944,384], fill=(45,80,130), width=5)
    _save_texture(TEXTURE_DIR/"target_wall.png", img)

    # Rubber paddle texture: red with small dimples.
    img = Image.new("RGB", (512, 512), (156, 18, 18)); draw = ImageDraw.Draw(img)
    for y in range(12,512,24):
        for x in range(12,512,24):
            draw.ellipse([x-3,y-3,x+3,y+3], fill=(105, 8, 8))
    _save_texture(TEXTURE_DIR/"paddle_rubber_red.png", img)

    # Handle texture.
    img = Image.new("RGB", (512, 512), (120, 78, 37)); draw = ImageDraw.Draw(img)
    for x in range(0,512,16):
        c=int(95 + rng.integers(-10,35)); draw.line([x,0,x+rng.integers(-6,7),512], fill=(c, int(c*0.62), int(c*0.32)), width=5)
    _save_texture(TEXTURE_DIR/"handle_wood.png", img)

    # Ball texture: off-white/orange with seam-like arcs.
    img = Image.new("RGB", (512,512), (255, 190, 55)); draw = ImageDraw.Draw(img)
    draw.arc([60,80,452,430], 25, 335, fill=(245,245,235), width=8)
    draw.arc([85,55,430,455], 205, 515, fill=(245,245,235), width=7)
    _save_texture(TEXTURE_DIR/"pingpong_ball.png", img)


def load_texture_assets(visual_asset="procedural"):
    ensure_visual_assets()
    assets = {p.name: p.read_bytes() for p in TEXTURE_DIR.glob("*.png")}
    if PROCEDURAL_MODEL_DIR.exists():
        for p in PROCEDURAL_MODEL_DIR.glob("*.obj"):
            assets[p.name] = p.read_bytes()
    if visual_asset == "sketchfab":
        mesh_path = EXTERNAL_MODEL_DIR / "sketchfab_paddle_mujoco_visual.obj"
        if not mesh_path.exists():
            raise FileNotFoundError(f"Missing Sketchfab visual mesh: {mesh_path}")
        assets[mesh_path.name] = mesh_path.read_bytes()
    return assets


def quat_mul(q1, q2):
    w1,x1,y1,z1=q1; w2,x2,y2,z2=q2
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2,
                     w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2,
                     w1*z2+x1*y2-y1*x2+z1*w2], dtype=float)


def axis_quat(axis, angle):
    axis=np.asarray(axis,dtype=float); axis=axis/np.linalg.norm(axis)
    s=math.sin(angle/2.0)
    return np.array([math.cos(angle/2.0), axis[0]*s, axis[1]*s, axis[2]*s], dtype=float)


def rpy_quat(roll, pitch, yaw):
    qx=axis_quat([1,0,0], roll); qy=axis_quat([0,1,0], pitch); qz=axis_quat([0,0,1], yaw)
    q=quat_mul(qz, quat_mul(qy, qx))
    return q/np.linalg.norm(q)


def _friction_text(values, default):
    vals = default if values is None else values
    return " ".join(f"{float(v):.7g}" for v in vals)


def _random_unit(rng):
    v = rng.normal(size=3)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return v / n


def _case_friction(case, key, default):
    if not case:
        return default
    return case.get(key, default)

def camera_axes(pos, target=(0.72, 0.0, 0.78), up=(0.0, 0.0, 1.0)):
    pos = np.asarray(pos, dtype=float)
    target = np.asarray(target, dtype=float)
    up = np.asarray(up, dtype=float)
    z_axis = pos - target
    z_axis = z_axis / max(1e-9, float(np.linalg.norm(z_axis)))
    x_axis = np.cross(up, z_axis)
    if float(np.linalg.norm(x_axis)) < 1e-7:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
    x_axis = x_axis / max(1e-9, float(np.linalg.norm(x_axis)))
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(1e-9, float(np.linalg.norm(y_axis)))
    return x_axis, y_axis


def camera_xml(name, pos, target=(0.72, 0.0, 0.78), fovy=42):
    x_axis, y_axis = camera_axes(pos, target)
    pos_txt = " ".join(f"{float(v):.6f}" for v in pos)
    xy_txt = " ".join(f"{float(v):.6f}" for v in [*x_axis, *y_axis])
    return f'    <camera name="{name}" pos="{pos_txt}" xyaxes="{xy_txt}" fovy="{float(fovy):.3g}"/>\n'


def _paddle_velocity_xyz(case):
    if "paddle_velocity_xyz" in case:
        return np.array(case["paddle_velocity_xyz"], dtype=float)
    return np.array(case.get("paddle_velocity_dir", [1,0,0]), dtype=float) * float(case["paddle_speed"])


def make_xml(case=None, visual_asset="procedural"):
    cameras_xml = "".join(
        [
            camera_xml("front", (0.78, -3.55, 1.42), fovy=42),
            camera_xml("left", (0.68, 3.30, 1.34), fovy=42),
            camera_xml("right", (0.68, -3.30, 1.34), fovy=42),
            camera_xml("top", (0.85, 0.00, 4.35), target=(0.85, 0.0, 0.78), fovy=34),
            camera_xml("left_oblique", (1.82, 2.70, 1.70), fovy=43),
            camera_xml("right_oblique", (1.82, -2.70, 1.70), fovy=43),
            camera_xml("overview", (1.00, -3.20, 1.55), fovy=43),
            camera_xml("wall", (1.25, -2.35, 1.05), fovy=36),
            camera_xml("side", (0.80, 2.80, 1.18), fovy=40),
            camera_xml("paddle", (-0.72, -1.28, 1.05), target=(0.25, 0.0, 0.85), fovy=38),
        ]
    )
    if visual_asset == "sketchfab":
        paddle_mesh_asset = '    <mesh name="sketchfab_paddle_visual_mesh" file="sketchfab_paddle_mujoco_visual.obj"/>\n'
        paddle_visual_geom = '      <geom name="paddle_sketchfab_visual_mesh" type="mesh" mesh="sketchfab_paddle_visual_mesh" material="rubber_red" contype="0" conaffinity="0" group="2"/>\n'
        paddle_face_attrs = 'rgba="0.90 0.05 0.05 0.0"'
        paddle_aux_visual_geoms = ''
    else:
        paddle_mesh_asset = '    <mesh name="procedural_paddle_visual_mesh" file="procedural_pingpong_paddle_visual.obj"/>\n'
        paddle_visual_geom = '      <geom name="paddle_visual_mesh" type="mesh" mesh="procedural_paddle_visual_mesh" material="rubber_red" contype="0" conaffinity="0" group="2"/>\n'
        paddle_face_attrs = 'material="rubber_red"'
        paddle_aux_visual_geoms = f'''      <geom name="paddle_back" type="cylinder" euler="0 1.57079632679 0" pos="-0.010 0 0" size="{PADDLE_RADIUS*0.96} {PADDLE_THICKNESS*0.35}" material="rubber_black" contype="0" conaffinity="0"/>\n      <geom name="paddle_handle" type="capsule" fromto="-0.02 0 -0.10 -0.02 0 -0.34" size="0.020" material="wood" contype="0" conaffinity="0"/>\n'''
    ball_friction = _friction_text(_case_friction(case, "ball_friction", DEFAULT_BALL_FRICTION), DEFAULT_BALL_FRICTION)
    paddle_friction = _friction_text(_case_friction(case, "paddle_friction", DEFAULT_PADDLE_FRICTION), DEFAULT_PADDLE_FRICTION)
    return f'''
<mujoco model="table_tennis_track_demo3">
  <compiler angle="radian" coordinate="local" inertiafromgeom="true"/>
  <option timestep="{DT}" gravity="0 0 -9.81" integrator="RK4" iterations="90" tolerance="1e-10"/>
  <visual><headlight ambient="0.34 0.34 0.34" diffuse="0.68 0.68 0.64" specular="0.12 0.12 0.12"/><global offwidth="1280" offheight="720"/><quality shadowsize="4096" offsamples="4"/><map znear="0.01" zfar="20"/></visual>
  <asset>
    <texture name="indoor_sky" type="skybox" builtin="gradient" rgb1="0.78 0.80 0.82" rgb2="0.50 0.56 0.62" width="512" height="512"/>
    <texture name="floor_tex" type="2d" file="wood_floor.png"/>
    <texture name="room_wall_tex" type="2d" file="indoor_wall.png"/>
    <texture name="target_wall_tex" type="2d" file="target_wall.png"/>
    <texture name="paddle_red_tex" type="2d" file="paddle_rubber_red.png"/>
    <texture name="handle_wood_tex" type="2d" file="handle_wood.png"/>
    <texture name="ball_tex" type="2d" file="pingpong_ball.png"/>
{paddle_mesh_asset}    <material name="floor_mat" texture="floor_tex" texrepeat="3 3" reflectance="0.22"/>
    <material name="room_wall_mat" texture="room_wall_tex" texrepeat="2 1" reflectance="0.12"/>
    <material name="ball_mat" texture="ball_tex" rgba="1 1 1 1" reflectance="0.18"/>
    <material name="rubber_red" texture="paddle_red_tex" rgba="1 1 1 1" reflectance="0.05"/>
    <material name="rubber_black" rgba="0.02 0.018 0.016 1" reflectance="0.08"/>
    <material name="wood" texture="handle_wood_tex" rgba="1 1 1 1" reflectance="0.12"/>
    <material name="wall_mat" texture="target_wall_tex" texrepeat="2 1" rgba="0.92 0.96 1 1" reflectance="0.10"/>
    <material name="target_panel_mat" texture="target_wall_tex" texrepeat="1 1" rgba="1 1 1 1" emission="0.16" reflectance="0.05"/>
    <material name="wall_frame" rgba="0.06 0.18 0.34 1" reflectance="0.18"/>
  </asset>
  <worldbody>
    <light name="key" pos="0 -3.6 4.8" dir="0.15 0.55 -1" diffuse="0.95 0.95 0.88" specular="0.25 0.25 0.25"/>
    <light name="fill" pos="2.5 1.8 3.2" dir="-0.7 -0.4 -1" diffuse="0.35 0.38 0.45"/>
{cameras_xml.rstrip()}
    <geom name="floor" type="plane" size="6 6 0.05" material="floor_mat" friction="0.8 0.02 0.001"/>
    <geom name="room_back_wall" type="box" pos="-1.15 0 1.25" size="0.035 3.2 1.25" material="room_wall_mat" contype="0" conaffinity="0"/>
    <geom name="room_left_wall" type="box" pos="0.95 2.65 1.25" size="3.3 0.035 1.25" material="room_wall_mat" contype="0" conaffinity="0"/>
    <geom name="wall" type="box" pos="{WALL_X} 0 {WALL_HEIGHT/2}" size="0.025 {WALL_WIDTH/2} {WALL_HEIGHT/2}" material="wall_mat" friction="0.28 0.01 0.001" solref="0.003 0.25" solimp="0.95 0.99 0.001"/>
    <geom name="wall_visual_panel" type="box" pos="{WALL_X-0.031} 0 {WALL_HEIGHT/2}" size="0.004 {WALL_WIDTH/2} {WALL_HEIGHT/2}" material="target_panel_mat" contype="0" conaffinity="0"/>
    <geom name="wall_top" type="box" pos="{WALL_X-0.01} 0 {WALL_HEIGHT}" size="0.04 {WALL_WIDTH/2} 0.012" material="wall_frame" contype="0" conaffinity="0"/>
    <body name="ball" pos="0.25 0 1.35">
      <freejoint name="ball_free"/>
      <geom name="ball_geom" type="sphere" size="{BALL_RADIUS}" mass="{BALL_MASS}" material="ball_mat" condim="4" friction="{ball_friction}" solref="0.0015 0.15" solimp="0.98 0.996 0.0004"/>
    </body>
    <body name="paddle" mocap="true" pos="-0.25 0 0.85" quat="1 0 0 0">
{paddle_visual_geom}      <geom name="paddle_face" type="cylinder" euler="0 1.57079632679 0" size="{PADDLE_RADIUS} {PADDLE_THICKNESS}" {paddle_face_attrs} condim="4" friction="{paddle_friction}" solref="0.0018 0.22" solimp="0.97 0.996 0.0004"/>
{paddle_aux_visual_geoms}    </body>
    <site name="wall_origin_marker" pos="{WALL_X} 0 0.85" size="0.014" rgba="0 0.25 1 1"/>
  </worldbody>
</mujoco>
'''


def choose_mode(case_id, mode):
    if mode != "balanced":
        return mode
    return ["target_wall", "ground_miss", "over_wall"][case_id % 3]


def parse_sample_ratios(text):
    if not text:
        return {"target_wall": 1.0, "ground_miss": 1.0, "over_wall": 1.0, "random": 0.35}
    ratios = {}
    for part in text.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"Bad --sample-ratios item {part!r}; expected name=value")
        k, v = part.split("=", 1)
        k = k.strip()
        if k not in {"target_wall", "ground_miss", "over_wall", "random"}:
            raise ValueError(f"Unknown sample ratio mode {k!r}")
        ratios[k] = float(v)
    return ratios or {"target_wall": 1.0, "ground_miss": 1.0, "over_wall": 1.0, "random": 0.35}


def draw_mode(rng, case_id, mode, ratios):
    if mode != "balanced":
        return choose_mode(case_id, mode)
    names = list(ratios.keys())
    weights = np.array([max(0.0, ratios[n]) for n in names], dtype=float)
    if float(weights.sum()) <= 0:
        return choose_mode(case_id, "balanced")
    weights /= weights.sum()
    return str(rng.choice(names, p=weights))


def sample_paddle_velocity(rng, mode):
    if mode == "target_wall":
        vx = rng.uniform(5.8, 9.4); vy = rng.uniform(-0.45, 0.45); vz = rng.uniform(-0.25, 1.10)
    elif mode == "over_wall":
        vx = rng.uniform(7.5, 12.0); vy = rng.uniform(-0.35, 0.35); vz = rng.uniform(1.7, 4.3)
    elif mode == "ground_miss":
        vx = rng.uniform(2.3, 5.0); vy = rng.uniform(-0.55, 0.55); vz = rng.uniform(-1.6, 0.15)
    else:
        vx = rng.uniform(2.2, 11.8); vy = rng.uniform(-1.15, 1.15); vz = rng.uniform(-1.8, 3.5)
    vel = np.array([vx, vy, vz], dtype=float)
    return vel, float(np.linalg.norm(vel))


def sample_friction(rng, profile):
    if profile == "low":
        ball_sliding = rng.uniform(0.015, 0.050); paddle_sliding = rng.uniform(0.25, 0.70)
    elif profile == "high":
        ball_sliding = rng.uniform(0.070, 0.140); paddle_sliding = rng.uniform(1.00, 1.80)
    else:
        ball_sliding = rng.uniform(0.025, 0.105); paddle_sliding = rng.uniform(0.45, 1.45)
    ball = [ball_sliding, rng.uniform(0.001, 0.012), rng.uniform(0.00003, 0.0007)]
    paddle = [paddle_sliding, rng.uniform(0.006, 0.055), rng.uniform(0.0002, 0.004)]
    return ball, paddle


def sample_case(rng, case_id, mode="balanced"):
    mode = choose_mode(case_id, mode)
    ball_x = rng.uniform(0.16, 0.46)
    ball_y = rng.uniform(-0.13, 0.13)
    ball_z0 = rng.uniform(1.12, 1.70)
    if mode == "target_wall":
        requested_hit_z = rng.uniform(0.82, 1.03); elev = math.radians(rng.uniform(-2, 8)); miss_y = rng.uniform(-0.008, 0.008)
    elif mode == "over_wall":
        requested_hit_z = rng.uniform(0.95, 1.16); elev = math.radians(rng.uniform(18, 36)); miss_y = rng.uniform(-0.008, 0.008)
    elif mode == "ground_miss":
        requested_hit_z = rng.uniform(0.70, 0.92); elev = math.radians(rng.uniform(-13, 1)); miss_y = rng.uniform(-0.015, 0.015)
    else:
        requested_hit_z = rng.uniform(0.72, 1.14); elev = math.radians(rng.uniform(-14, 34)); miss_y = rng.uniform(-0.030, 0.030)
    requested_hit_z = min(requested_hit_z, ball_z0 - 0.06)
    t_hit = math.sqrt(max(0.02, 2.0*(ball_z0-requested_hit_z)/G)) + rng.uniform(-0.018, 0.018)
    t_hit = max(0.16, min(t_hit, math.sqrt(max(0.02, 2.0*(ball_z0-0.58)/G))))
    hit_z = ball_z0 - 0.5 * G * t_hit * t_hit
    velocity, speed = sample_paddle_velocity(rng, mode)
    # Keep the planned path coherent with sampled 3D velocity. The elevation sample
    # still influences pitch, while the actual impact vector contains x/y/z speed.
    velocity[2] = 0.55 * velocity[2] + 0.45 * (math.tan(elev) * max(0.5, velocity[0]))
    speed = float(np.linalg.norm(velocity))
    dir_vec = velocity / max(1e-9, speed)
    roll = math.radians(rng.uniform(-18, 18))
    pitch = -math.atan2(velocity[2], max(0.4, velocity[0])) + math.radians(rng.uniform(-10, 10))
    yaw = math.atan2(velocity[1], max(0.4, velocity[0])) + math.radians(rng.uniform(-18, 18))
    paddle_center_hit = np.array([ball_x - BALL_RADIUS - PADDLE_THICKNESS*1.15, ball_y + miss_y, hit_z], dtype=float)
    timing_jitter = rng.uniform(-0.035, 0.035)
    move_duration = max(1e-4, t_hit)
    t_start = 0.0
    travel = float(np.clip(speed * move_duration, 0.24, 0.86))
    paddle_start = paddle_center_hit - dir_vec * travel
    spin_axis = _random_unit(rng)
    if mode == "target_wall":
        spin_speed = rng.uniform(0.0, 240.0)
    elif mode == "over_wall":
        spin_speed = rng.uniform(60.0, 420.0)
    elif mode == "ground_miss":
        spin_speed = rng.uniform(0.0, 320.0)
    else:
        spin_speed = rng.uniform(0.0, 520.0)
    ball_angular_vel = spin_axis * spin_speed
    friction_profile = rng.choice(["low", "mid", "high"], p=[0.22, 0.56, 0.22])
    ball_friction, paddle_friction = sample_friction(rng, friction_profile)
    ball_initial_vel = np.array([rng.uniform(-0.06, 0.06), rng.uniform(-0.045, 0.045), rng.uniform(-0.04, 0.015)], dtype=float)
    return dict(episode_id=int(case_id), sample_mode=mode, ball_initial_pos=[float(ball_x), float(ball_y), float(ball_z0)], ball_initial_vel=ball_initial_vel.astype(float).tolist(),
                hit_z=float(hit_z), planned_t_hit=float(t_hit), move_t_start=float(t_start), move_duration=float(move_duration), paddle_speed=float(speed),
                paddle_velocity_dir=dir_vec.astype(float).tolist(), paddle_start=paddle_start.astype(float).tolist(), paddle_hit_pose=paddle_center_hit.astype(float).tolist(),
                paddle_velocity_xyz=velocity.astype(float).tolist(), paddle_roll=float(roll), paddle_pitch=float(pitch), paddle_yaw=float(yaw),
                ball_spin_axis=spin_axis.astype(float).tolist(), ball_spin_speed_rad_s=float(spin_speed), ball_initial_angular_vel=ball_angular_vel.astype(float).tolist(),
                friction_profile=str(friction_profile), ball_friction=[float(x) for x in ball_friction], paddle_friction=[float(x) for x in paddle_friction],
                drop_height_m=float(ball_z0), drop_distance_x_m=float(ball_x), drop_lateral_y_m=float(ball_y), timing_jitter_s=float(timing_jitter))


def paddle_pose(case, t):
    p0=np.array(case["paddle_start"], dtype=float); ph=np.array(case["paddle_hit_pose"], dtype=float)
    t0=case["move_t_start"]; th=case["planned_t_hit"]; T=max(1e-4, th-t0)
    vhit=np.array(case.get("paddle_velocity_dir", [1,0,0]), dtype=float) * case["paddle_speed"]
    if t <= t0:
        pos=p0
    elif t <= th:
        tau=t-t0
        s=np.clip(tau/T, 0.0, 1.0)
        pos=p0 + (ph-p0)*s
    else:
        pos=ph
    return pos, rpy_quat(case["paddle_roll"], case["paddle_pitch"], case["paddle_yaw"])


def set_ball_state(model, data, pos, vel, angular_vel=None):
    jid=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free"); qadr=model.jnt_qposadr[jid]; vadr=model.jnt_dofadr[jid]
    w = [0,0,0] if angular_vel is None else angular_vel
    data.qpos[qadr:qadr+7]=[pos[0],pos[1],pos[2],1,0,0,0]; data.qvel[vadr:vadr+6]=[vel[0],vel[1],vel[2],w[0],w[1],w[2]]


def get_ball_state(model, data):
    jid=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free"); qadr=model.jnt_qposadr[jid]; vadr=model.jnt_dofadr[jid]
    return data.qpos[qadr:qadr+3].copy(), data.qvel[vadr:vadr+3].copy(), data.qvel[vadr+3:vadr+6].copy()


def overlay_frame(frame, result, t, camera, trail_len=0):
    im = Image.fromarray(frame)
    draw = ImageDraw.Draw(im)
    status = "HIT" if result.get("success") else (result.get("miss_reason") or "running")
    color = (40, 230, 90) if result.get("success") else (255, 210, 40)
    lines = [f"Demo3 TableTennisTrack | {camera}", f"mode={result.get('sample_mode')}  t={t:0.3f}s  status={status}", f"wall x={WALL_X:.2f} height={WALL_HEIGHT:.2f}m"]
    if result.get("wall_hit_local") is not None:
        local = result["wall_hit_local"]; world = result["wall_hit_world"]
        lines.append(f"wall local(right,up)=({local[0]:+.3f}, {local[1]:+.3f}) m")
        lines.append(f"wall world=({world[0]:.3f}, {world[1]:+.3f}, {world[2]:.3f}) m")
    else:
        lines.append("wall hit: none / miss")
    x0, y0 = 18, 16
    draw.rounded_rectangle([10, 8, 560, 120], radius=8, fill=(0,0,0,150))
    for k, line in enumerate(lines):
        draw.text((x0, y0+20*k), line, fill=color if k == 1 else (245,245,245))
    if trail_len:
        draw.text((18, 126), f"recorded ball trajectory samples: {trail_len}", fill=(220, 235, 255))
    return np.asarray(im)


def run_episode(case, out_dir=None, render=False, camera="overview", width=1280, height=720, video_tag=None, save_npz=True, visual_asset="procedural"):
    model=mujoco.MjModel.from_xml_string(make_xml(case, visual_asset=visual_asset), assets=load_texture_assets(visual_asset)); data=mujoco.MjData(model); mujoco.mj_resetData(model, data)
    set_ball_state(model, data, case["ball_initial_pos"], case["ball_initial_vel"], case.get("ball_initial_angular_vel", [0,0,0]))
    p,q=paddle_pose(case, 0.0); data.mocap_pos[0]=p; data.mocap_quat[0]=q; mujoco.mj_forward(model, data)
    wall_gid=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "wall"); ball_gid=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom"); paddle_gid=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "paddle_face")
    renderer=None; raw_frames=[]; frame_times=[]; render_every=8
    if render:
        renderer=mujoco.Renderer(model, height=height, width=width)
    rows=[]; contacts=[]; hit=None; first_paddle_contact=None; miss_reason=None; min_ball_wall_abs=999.0

    def scan_ball_contacts(t, paddle_pos):
        nonlocal hit, first_paddle_contact
        for ci in range(data.ncon):
            c=data.contact[ci]; gs={int(c.geom1), int(c.geom2)}
            if ball_gid in gs:
                other_gid = int(c.geom2) if int(c.geom1) == ball_gid else int(c.geom1)
                other_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other_gid) or f"geom_{other_gid}"
                if other_gid == paddle_gid or other_name.startswith("paddle"):
                    ev={"t":float(t), "type":"ball_paddle", "geom":other_name, "pos":np.array(c.pos).astype(float).tolist()}
                    contacts.append(ev)
                    if first_paddle_contact is None:
                        first_paddle_contact=ev
                if other_gid == wall_gid and hit is None:
                    hp=np.array(c.pos, dtype=float); origin_y=paddle_pos[1]; origin_z=paddle_pos[2]
                    hit={"t":float(t), "world":hp.tolist(), "local":[float(hp[1]-origin_y), float(hp[2]-origin_z)]}

    render_tail_s = 0.85 if render else 0.18
    stop_after_t = None
    for i in range(int(MAX_TIME/DT)+1):
        t=i*DT; p,q=paddle_pose(case, t); data.mocap_pos[0]=p; data.mocap_quat[0]=q
        mujoco.mj_forward(model, data)
        scan_ball_contacts(t, p)
        mujoco.mj_step(model, data)
        scan_ball_contacts(t, p)
        bpos,bvel,bangvel=get_ball_state(model, data)
        min_ball_wall_abs = min(min_ball_wall_abs, abs(float(WALL_X - bpos[0])))
        if i % TRAJ_STRIDE == 0:
            rows.append([t, *bpos.tolist(), *bvel.tolist(), *bangvel.tolist(), *p.tolist(), *q.tolist()])
        if render and i % render_every == 0:
            renderer.update_scene(data, camera=camera); raw_frames.append(renderer.render().copy()); frame_times.append(t)
        if hit is not None and stop_after_t is None:
            stop_after_t = hit["t"] + render_tail_s
        if stop_after_t is not None and t > stop_after_t:
            break
        if hit is None and miss_reason is None:
            if bpos[2] < BALL_RADIUS*0.55 and t > 0.05:
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
    traj=np.array(rows, dtype=np.float64)
    success=hit is not None and 0.0 <= hit["world"][2] <= WALL_HEIGHT
    if not success and miss_reason is None:
        miss_reason="wall_contact_outside_height" if hit is not None else "unknown_miss"
    result={**case, "success":bool(success), "wall_hit_world":hit["world"] if hit else None, "wall_hit_local":hit["local"] if hit else None, "wall_hit_t":hit["t"] if hit else None,
            "miss_reason":None if success else miss_reason, "first_paddle_contact":first_paddle_contact, "contact_events":contacts[:48], "min_ball_wall_abs":float(min_ball_wall_abs),
            "wall_x":WALL_X, "wall_height":WALL_HEIGHT, "dt":DT, "trajectory_hz":int(1/(DT*TRAJ_STRIDE)),
            "trajectory_columns":["t","ball_x","ball_y","ball_z","ball_vx","ball_vy","ball_vz","ball_wx","ball_wy","ball_wz","paddle_x","paddle_y","paddle_z","paddle_qw","paddle_qx","paddle_qy","paddle_qz"]}
    if out_dir is not None:
        out=Path(out_dir); out.mkdir(parents=True, exist_ok=True); eid=f"episode_{case['episode_id']:06d}"
        control = np.array([
            case["paddle_speed"],
            *_paddle_velocity_xyz(case).tolist(),
            case["paddle_roll"], case["paddle_pitch"], case["paddle_yaw"],
            case["move_t_start"], case["planned_t_hit"], case["move_duration"],
            *case.get("ball_friction", DEFAULT_BALL_FRICTION),
            *case.get("paddle_friction", DEFAULT_PADDLE_FRICTION),
            *case.get("ball_spin_axis", [0,0,1]), case.get("ball_spin_speed_rad_s", 0.0),
            *case.get("ball_initial_angular_vel", [0,0,0]),
            *case["ball_initial_pos"],
        ], dtype=np.float64)
        if save_npz:
            np.savez_compressed(
                out/f"{eid}.npz",
                trajectory=traj,
                metadata=json.dumps(result, ensure_ascii=False),
                paddle_control=control,
                paddle_control_columns=np.array([
                    "paddle_speed_m_s",
                    "paddle_vx_m_s", "paddle_vy_m_s", "paddle_vz_m_s",
                    "paddle_roll_rad", "paddle_pitch_rad", "paddle_yaw_rad",
                    "paddle_move_start_time_s", "paddle_planned_hit_time_s", "paddle_move_duration_s",
                    "ball_friction_sliding", "ball_friction_torsional", "ball_friction_rolling",
                    "paddle_friction_sliding", "paddle_friction_torsional", "paddle_friction_rolling",
                    "ball_spin_axis_x", "ball_spin_axis_y", "ball_spin_axis_z", "ball_spin_speed_rad_s",
                    "ball_initial_wx_rad_s", "ball_initial_wy_rad_s", "ball_initial_wz_rad_s",
                    "ball_initial_x_m", "ball_initial_y_m", "ball_initial_z_m",
                ]),
                paddle_start_pose=np.array([*case["paddle_start"], *rpy_quat(case["paddle_roll"], case["paddle_pitch"], case["paddle_yaw"])], dtype=np.float64),
                paddle_hit_pose=np.array([*case["paddle_hit_pose"], *rpy_quat(case["paddle_roll"], case["paddle_pitch"], case["paddle_yaw"])], dtype=np.float64),
                paddle_velocity_dir=np.array(case["paddle_velocity_dir"], dtype=np.float64),
                paddle_velocity_xyz=_paddle_velocity_xyz(case).astype(np.float64),
                ball_initial_pos=np.array(case["ball_initial_pos"], dtype=np.float64),
                ball_initial_vel=np.array(case["ball_initial_vel"], dtype=np.float64),
                ball_initial_angular_vel=np.array(case.get("ball_initial_angular_vel", [0,0,0]), dtype=np.float64),
                ball_spin_axis=np.array(case.get("ball_spin_axis", [0,0,1]), dtype=np.float64),
                ball_spin_speed_rad_s=np.array([case.get("ball_spin_speed_rad_s", 0.0)], dtype=np.float64),
                ball_friction=np.array(case.get("ball_friction", DEFAULT_BALL_FRICTION), dtype=np.float64),
                paddle_friction=np.array(case.get("paddle_friction", DEFAULT_PADDLE_FRICTION), dtype=np.float64),
            )
        if render and raw_frames:
            tagged = video_tag or camera
            frames=[overlay_frame(f, result, ft, camera, len(traj)) for f, ft in zip(raw_frames, frame_times)]
            imageio.mimsave(out/f"{eid}_{tagged}.mp4", frames, fps=int(1/(DT*render_every)), quality=9, macro_block_size=16)
    if renderer is not None:
        renderer.close()
    return result


def write_summary(out, summary):
    succ=sum(1 for r in summary if r["success"]); reasons={}; modes={}
    for r in summary:
        modes.setdefault(r["sample_mode"], {"n":0,"success":0,"miss":{}})
        modes[r["sample_mode"]]["n"] += 1
        if r["success"]:
            modes[r["sample_mode"]]["success"] += 1
        else:
            reason=r["miss_reason"]; reasons[reason]=reasons.get(reason,0)+1; modes[r["sample_mode"]]["miss"][reason]=modes[r["sample_mode"]]["miss"].get(reason,0)+1
    report={"episodes":len(summary),"success":succ,"success_rate":succ/max(1,len(summary)),"miss_reasons":reasons,"by_sample_mode":modes,"output_dir":str(Path(out).resolve()),"created_at":time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    Path(out,"summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


def load_existing_result(npz_path):
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            return json.loads(str(data["metadata"]))
    except Exception as exc:
        return {"episode_id": int(Path(npz_path).stem.split("_")[-1]), "success": False, "sample_mode": "unknown", "miss_reason": f"metadata_load_failed:{exc}"}


def index_record(result, out):
    eid = int(result["episode_id"])
    return {
        "episode_id": eid,
        "episode_npz": str((Path(out) / f"episode_{eid:06d}.npz").resolve()),
        "sample_mode": result.get("sample_mode"),
        "success": bool(result.get("success")),
        "miss_reason": result.get("miss_reason"),
        "wall_hit_world": result.get("wall_hit_world"),
        "wall_hit_local": result.get("wall_hit_local"),
        "wall_hit_t": result.get("wall_hit_t"),
        "ball_initial_pos": result.get("ball_initial_pos"),
        "ball_initial_vel": result.get("ball_initial_vel"),
        "ball_initial_angular_vel": result.get("ball_initial_angular_vel"),
        "ball_spin_axis": result.get("ball_spin_axis"),
        "ball_spin_speed_rad_s": result.get("ball_spin_speed_rad_s"),
        "ball_friction": result.get("ball_friction"),
        "paddle_friction": result.get("paddle_friction"),
        "paddle_velocity_xyz": result.get("paddle_velocity_xyz"),
        "paddle_roll": result.get("paddle_roll"),
        "paddle_pitch": result.get("paddle_pitch"),
        "paddle_yaw": result.get("paddle_yaw"),
        "move_t_start": result.get("move_t_start"),
        "planned_t_hit": result.get("planned_t_hit"),
        "move_duration": result.get("move_duration"),
        "trajectory_columns": result.get("trajectory_columns"),
    }


def write_manifest_and_index(out, results):
    out = Path(out)
    with (out / "manifest.jsonl").open("w", encoding="utf-8") as f_manifest, (out / "dataset_index.jsonl").open("w", encoding="utf-8") as f_index:
        for r in sorted(results, key=lambda x: int(x.get("episode_id", 0))):
            f_manifest.write(json.dumps(r, ensure_ascii=False) + "\n")
            f_index.write(json.dumps(index_record(r, out), ensure_ascii=False) + "\n")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=60); ap.add_argument("--seed", type=int, default=7); ap.add_argument("--out", type=str, default="Outputs/balanced")
    ap.add_argument("--videos", type=int, default=6); ap.add_argument("--camera", type=str, default="front"); ap.add_argument("--video-cameras", type=str, default="front,left,right,top,left_oblique,right_oblique")
    ap.add_argument("--video-stride", type=int, default=0, help="if >0, render every Nth episode until --videos is reached; otherwise render first --videos")
    ap.add_argument("--start-id", type=int, default=0, help="first episode id for resumable large-scale jobs")
    ap.add_argument("--resume", action="store_true", help="skip existing episode_XXXXXX.npz and reuse its metadata")
    ap.add_argument("--sample-ratios", type=str, default="", help="balanced-mode ratios, e.g. target_wall=4,ground_miss=2,over_wall=2,random=1")
    ap.add_argument("--progress-every", type=int, default=100, help="print progress every N episodes")
    ap.add_argument("--mode", choices=["balanced","target_wall","ground_miss","over_wall","random"], default="balanced")
    ap.add_argument("--showcase", action="store_true", help="render representative multi-view videos")
    ap.add_argument("--visual-asset", choices=["procedural", "sketchfab"], default="procedural", help="visual-only paddle asset; physics uses calibrated MuJoCo primitives")
    args=ap.parse_args(); rng=np.random.default_rng(args.seed); out=Path(args.out); out.mkdir(parents=True, exist_ok=True); summary=[]
    cameras=[c.strip() for c in args.video_cameras.split(',') if c.strip()]
    ratios=parse_sample_ratios(args.sample_ratios)
    cases=[]
    for ep in range(args.start_id, args.start_id + args.episodes):
        case_mode = draw_mode(rng, ep, args.mode, ratios)
        cases.append(sample_case(rng, ep, case_mode))
    for idx, case in enumerate(cases, start=1):
        npz_path = out / f"episode_{case['episode_id']:06d}.npz"
        if args.resume and npz_path.exists():
            res = load_existing_result(npz_path)
        else:
            res=run_episode(case, out, render=False, camera=args.camera, save_npz=True, visual_asset=args.visual_asset)
        summary.append(res)
        if args.progress_every > 0 and (idx % args.progress_every == 0 or idx == len(cases)):
            succ=sum(1 for r in summary if r.get("success"))
            print(f"[progress] {idx}/{len(cases)} episodes, success_rate={succ/max(1,idx):.3f}, out={out}", flush=True)
    write_manifest_and_index(out, summary)
    # Multi-view video采集: render selected episodes only; keep one NPZ per episode.
    if args.video_stride > 0:
        video_cases = [c for idx, c in enumerate(cases) if idx % args.video_stride == 0][:max(0,args.videos)]
    else:
        video_cases = cases[:max(0,args.videos)]
    for case in video_cases:
        for cam in cameras:
            run_episode(case, out, render=True, camera=cam, video_tag=cam, save_npz=False, visual_asset=args.visual_asset)
    if args.showcase:
        picked=[]; seen=set()
        for case in cases:
            if case["sample_mode"] not in seen:
                picked.append(case); seen.add(case["sample_mode"])
            if len(picked) >= 3:
                break
        for case in picked:
            c2=dict(case); c2["episode_id"]=100000+case["episode_id"]
            for cam in cameras:
                run_episode(c2, out, render=True, camera=cam, video_tag=f"showcase_{case['sample_mode']}_{cam}", save_npz=False, visual_asset=args.visual_asset)
    write_summary(out, summary)

if __name__ == "__main__":
    main()
