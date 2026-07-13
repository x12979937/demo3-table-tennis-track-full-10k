#!/usr/bin/env bash
set -Eeuo pipefail

TOTAL_EPISODES=${TOTAL_EPISODES:-10000}
START_EPISODE=${START_EPISODE:-auto}
WORKERS=${WORKERS:-2}
WORKER_BATCH_SIZE=${WORKER_BATCH_SIZE:-10}
SEED0=${SEED0:-2026071300}
WIDTH=${WIDTH:-1280}
HEIGHT=${HEIGHT:-720}
CAMERAS=${CAMERAS:-front,left,right,top,left_oblique,right_oblique,overview}
GPU_LIST=${GPU_LIST:-2,3}
PUSH_EVERY_BATCH=${PUSH_EVERY_BATCH:-1}
EXPECTED_VIEWS=7

FS_ROOT=/autodl-fs/data/mingyu/demo3_table_tennis_track_full_10k
TMP_ROOT=/root/autodl-tmp/mingyu/demo3_table_tennis_track_full_10k_work_parallel
PY=/root/autodl-tmp/mingyu/IsaacSim/py311isaac/bin/python
SCRIPT=$FS_ROOT/scripts/table_tennis_track_full_dataset.py
VERIFY=$FS_ROOT/scripts/verify_state_replay.py
ARCHIVE_DIR=$FS_ROOT/archives
META_DIR=$FS_ROOT/metadata
LOG_DIR=$FS_ROOT/logs
REPO_DIR=$FS_ROOT/github_repo
TOKEN_FILE=$FS_ROOT/.github_token
ASKPASS=$FS_ROOT/github_askpass.sh
STATUS=$FS_ROOT/status.json
PAR_STATUS=$FS_ROOT/parallel_status.json
PIPELINE_LOG=$LOG_DIR/pipeline_parallel.log
REPO_SLUG=x12979937/demo3-table-tennis-track-full-10k
REPO_URL=https://github.com/$REPO_SLUG.git
LOCK_DIR=$FS_ROOT/locks
CLAIM_FILE=$META_DIR/parallel_next_episode.txt

case "$FS_ROOT" in /autodl-fs/data/mingyu/*) ;; *) echo "unsafe FS_ROOT=$FS_ROOT" >&2; exit 10;; esac
case "$TMP_ROOT" in /root/autodl-tmp/mingyu/*|/root/autodl-tmp/data/mingyu/*) ;; *) echo "unsafe TMP_ROOT=$TMP_ROOT" >&2; exit 11;; esac
mkdir -p "$ARCHIVE_DIR" "$META_DIR" "$LOG_DIR" "$TMP_ROOT/batches" "$TMP_ROOT/tmp" "$FS_ROOT/git_tmp" "$FS_ROOT/git_home" "$LOCK_DIR"

log() { printf '[%s] %s\n' "$(date '+%F %T %z')" "$*" | tee -a "$PIPELINE_LOG"; }
source_network_turbo() { if [ -f /etc/network_turbo ]; then source /etc/network_turbo >/dev/null 2>&1 || true; fi; }

with_lock() {
  local lock_name=$1; shift
  local lock_file="$LOCK_DIR/$lock_name.lock"
  (
    flock 9
    "$@"
  ) 9>"$lock_file"
}

disk_json() {
  python3 - <<'PY'
import json, subprocess
rows=[]
for mount in ['/root/autodl-tmp','/autodl-fs/data']:
    try:
        line=subprocess.check_output(['df','-Pk',mount], text=True).splitlines()[1].split()
        inode=subprocess.check_output(['df','-Pi',mount], text=True).splitlines()[1].split()
        rows.append({'mount':line[-1],'avail_kb':int(line[3]),'used_pct':line[4],'inode_free':int(inode[3]),'inode_used_pct':inode[4]})
    except Exception as e:
        rows.append({'mount':mount,'error':str(e)})
print(','.join(json.dumps(r,separators=(',',':')) for r in rows))
PY
}

archived_count() {
  if [ -f "$META_DIR/archive_index.jsonl" ]; then
    python3 - "$META_DIR/archive_index.jsonl" <<'PY'
import json, sys
total=0
for line in open(sys.argv[1], encoding='utf-8'):
    line=line.strip()
    if not line: continue
    try: total += int(json.loads(line).get('episode_count', 0))
    except Exception: pass
print(total)
PY
  else echo 0; fi
}

first_missing_episode() {
  python3 - "$META_DIR/archive_index.jsonl" "$TOTAL_EPISODES" "$START_EPISODE" <<'PY'
import json, sys
p,total,start_arg=sys.argv[1],int(sys.argv[2]),sys.argv[3]
cur=0 if start_arg == 'auto' else int(start_arg)
intervals=[]
try:
    for line in open(p, encoding='utf-8'):
        line=line.strip()
        if not line: continue
        try:
            d=json.loads(line); intervals.append((int(d.get('start_episode',-1)), int(d.get('end_episode',-1))))
        except Exception: pass
except FileNotFoundError:
    pass
for s,e in sorted(intervals):
    if e < cur: continue
    if s > cur: break
    cur=max(cur, e+1)
print(min(cur,total))
PY
}

write_status() {
  local phase=${1:-running}; local current=${2:-0}; local message=${3:-parallel}; local count disks
  count=$(archived_count); disks=$(disk_json)
  cat > "$STATUS.tmp" <<EOF
{"updated_at":"$(date -Iseconds)","phase":"$phase","current_episode":$current,"total_episodes":$TOTAL_EPISODES,"archived_episode_count":$count,"batch_size":$WORKER_BATCH_SIZE,"workers":$WORKERS,"tmp_root":"$TMP_ROOT","fs_root":"$FS_ROOT","github_repo":"https://github.com/$REPO_SLUG","message":"$message","disks":[$disks]}
EOF
  mv "$STATUS.tmp" "$STATUS"
  cp -f "$STATUS" "$PAR_STATUS" 2>/dev/null || true
}

write_worker_status() {
  local wid=$1 phase=$2 start=$3 end=$4 msg=$5
  cat > "$FS_ROOT/status_worker_${wid}.json.tmp" <<EOF
{"updated_at":"$(date -Iseconds)","worker":$wid,"phase":"$phase","start_episode":$start,"end_episode":$end,"message":"$msg"}
EOF
  mv "$FS_ROOT/status_worker_${wid}.json.tmp" "$FS_ROOT/status_worker_${wid}.json"
}

cleanup_tmp() { find "$TMP_ROOT/tmp" -mindepth 1 -maxdepth 1 -print0 2>/dev/null | xargs -0r rm -rf --; }
require_tmp_space() {
  cleanup_tmp
  local avail; avail=$(df -Pk /root/autodl-tmp | awk 'NR==2{print $4}')
  if (( avail < 4194304 )); then log "tmp free space below 4GB: ${avail}KB"; write_status blocked "$(first_missing_episode)" tmp_free_below_4GB; exit 20; fi
}

claim_batch_locked() {
  local start end init
  if [ ! -f "$CLAIM_FILE" ]; then
    init=$(first_missing_episode)
    echo "$init" > "$CLAIM_FILE"
  fi
  start=$(cat "$CLAIM_FILE")
  if (( start >= TOTAL_EPISODES )); then return 1; fi
  end=$((start + WORKER_BATCH_SIZE - 1))
  if (( end >= TOTAL_EPISODES )); then end=$((TOTAL_EPISODES - 1)); fi
  echo $((end + 1)) > "$CLAIM_FILE"
  printf '%s %s\n' "$start" "$end"
}

claim_batch() {
  (
    flock 9
    claim_batch_locked
  ) 9>"$LOCK_DIR/claim.lock"
}

ensure_repo_public() {
  [ -s "$TOKEN_FILE" ] || { log "missing github token"; return 1; }
  source_network_turbo
  local token api out rc
  token=$(cat "$TOKEN_FILE")
  api=https://api.github.com
  out=$FS_ROOT/git_tmp/github_repo_api.json
  rc=0
  curl -fsS -H "Authorization: Bearer $token" -H "Accept: application/vnd.github+json" "$api/repos/$REPO_SLUG" -o "$out" >/dev/null 2>&1 || rc=$?
  if [ "$rc" != 0 ]; then
    curl -fsS -X POST -H "Authorization: Bearer $token" -H "Accept: application/vnd.github+json" "$api/user/repos" -d '{"name":"demo3-table-tennis-track-full-10k","private":false,"description":"Replayable MuJoCo table-tennis paddle-ball-wall 10k dataset with meshes, masks, bboxes, multiview RGB, and state validation."}' -o "$out" >/dev/null 2>&1 || true
  fi
  curl -fsS -X PATCH -H "Authorization: Bearer $token" -H "Accept: application/vnd.github+json" "$api/repos/$REPO_SLUG" -d '{"private":false}' -o "$out" >/dev/null 2>&1 || true
}

prepare_repo() {
  [ -x "$ASKPASS" ] || { log "missing github askpass"; return 1; }
  ensure_repo_public || return 1
  export GIT_ASKPASS="$ASKPASS" GIT_TERMINAL_PROMPT=0 HOME="$FS_ROOT/git_home" TMPDIR="$FS_ROOT/git_tmp"
  mkdir -p "$HOME" "$TMPDIR"
  if [ ! -d "$REPO_DIR/.git" ]; then
    rm -rf -- "$REPO_DIR"
    git clone "$REPO_URL" "$REPO_DIR" >> "$LOG_DIR/github_push.log" 2>&1 || { mkdir -p "$REPO_DIR"; git -C "$REPO_DIR" init >> "$LOG_DIR/github_push.log" 2>&1; git -C "$REPO_DIR" checkout -B main >> "$LOG_DIR/github_push.log" 2>&1; git -C "$REPO_DIR" remote add origin "$REPO_URL"; }
  fi
  git -C "$REPO_DIR" config user.name x12979937
  git -C "$REPO_DIR" config user.email mingyu_xu9646@163.com
  git -C "$REPO_DIR" remote set-url origin "$REPO_URL"
}

write_repo_files() {
  cat > "$META_DIR/generation_config.json" <<EOF
{"schema":"demo3_table_tennis_track_full_10k_strict_parallel_v1","total_episodes":$TOTAL_EPISODES,"worker_batch_size":$WORKER_BATCH_SIZE,"workers":$WORKERS,"width":$WIDTH,"height":$HEIGHT,"cameras":"$CAMERAS","engine":"MuJoCo","per_episode_validation":"state_replay_validation.json","contains_raw_rgb_frames":true,"contains_2d_bboxes":true,"contains_pixel_masks":true,"contains_episode_mesh_assets":true,"state_replay":"MuJoCo XML plus per-sample qpos/qvel/mocap state, validated after each episode before archive","parallelization":"workers claim non-overlapping episode ranges; archive and git operations are locked"}
EOF
  cat > "$FS_ROOT/README.md" <<'EOF'
# demo3 table tennis track full 10k

Strict regenerated MuJoCo dataset. Each archive contains episode folders with state replay validation, trajectory/state arrays, seven camera videos, raw RGB frame archives, perception files with 2D boxes and pixel masks, mesh assets, and plain plus structured descriptions.

Parallel generation uses non-overlapping episode claims and locked archive/GitHub updates.
EOF
}

sync_repo_and_push_locked() {
  local reason=${1:-batch} push_rc attempt
  write_repo_files
  prepare_repo || { log "github prepare failed for $reason"; return 1; }
  rm -rf -- "$REPO_DIR/archives" "$REPO_DIR/metadata" "$REPO_DIR/scripts" "$REPO_DIR/assets"
  mkdir -p "$REPO_DIR/archives" "$REPO_DIR/metadata" "$REPO_DIR/scripts" "$REPO_DIR/assets"
  (
    flock 8
    find "$ARCHIVE_DIR" -maxdepth 1 -type f \( -name '*.tar.gz' -o -name '*.tar.gz.part[0-9][0-9][0-9]' -o -name '*.sha256' -o -name '*.parts.json' -o -name '*.parts.sha256' \) -print0 | sort -z | while IFS= read -r -d '' f; do
      ln -f "$f" "$REPO_DIR/archives/$(basename "$f")" 2>/dev/null || cp -f "$f" "$REPO_DIR/archives/$(basename "$f")"
    done
    cp -f "$META_DIR/archive_index.jsonl" "$REPO_DIR/metadata/archive_index.jsonl" 2>/dev/null || : > "$REPO_DIR/metadata/archive_index.jsonl"
  ) 8>"$LOCK_DIR/archive.lock"
  cp -f "$META_DIR/generation_config.json" "$REPO_DIR/metadata/generation_config.json"
  cp -f "$STATUS" "$REPO_DIR/status.json" 2>/dev/null || true
  cp -f "$FS_ROOT/README.md" "$REPO_DIR/README.md"
  cp -f "$SCRIPT" "$REPO_DIR/scripts/table_tennis_track_full_dataset.py"
  cp -f "$VERIFY" "$REPO_DIR/scripts/verify_state_replay.py"
  cp -f "$FS_ROOT/scripts/table_tennis_track.py" "$REPO_DIR/scripts/table_tennis_track.py" 2>/dev/null || true
  cp -f "$FS_ROOT/run_demo3_10k_parallel.sh" "$REPO_DIR/scripts/run_demo3_10k_parallel.sh" 2>/dev/null || true
  [ -d "$FS_ROOT/assets" ] && cp -a "$FS_ROOT/assets/." "$REPO_DIR/assets/" 2>/dev/null || true
  git -C "$REPO_DIR" add -A README.md status.json archives metadata scripts assets >> "$LOG_DIR/github_push.log" 2>&1
  if git -C "$REPO_DIR" diff --cached --quiet; then log "github no changes for $reason"; return 0; fi
  git -C "$REPO_DIR" commit -m "Update strict demo3 replayable dataset ($reason)" >> "$LOG_DIR/github_push.log" 2>&1 || true
  source_network_turbo
  for attempt in 1 2 3 4 5; do
    push_rc=0
    git -C "$REPO_DIR" push origin HEAD:main >> "$LOG_DIR/github_push.log" 2>&1 || push_rc=$?
    if [ "$push_rc" = 0 ]; then log "github push ok for $reason"; return 0; fi
    log "github push attempt $attempt failed rc=$push_rc for $reason"; sleep $((attempt*20))
  done
  return 1
}

sync_repo_and_push() {
  local reason=${1:-batch}
  (
    flock 9
    sync_repo_and_push_locked "$reason"
  ) 9>"$LOCK_DIR/git.lock"
}

validate_episode_dir() {
  local ep_dir=$1 eid=$2
  "$PY" - "$ep_dir" "$eid" "$EXPECTED_VIEWS" <<'PY'
import gzip, json, sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]); eid=int(sys.argv[2]); views=int(sys.argv[3]); stem=f"episode_{eid:06d}"
required=[root/f"{stem}.npz", root/"state_replay_validation.json", root/f"{stem}_descriptions.json", root/f"{stem}_meshes/mesh_manifest.json"]
for p in required:
    assert p.is_file() and p.stat().st_size>0, str(p)
val=json.loads((root/"state_replay_validation.json").read_text(encoding="utf-8"))
assert val.get("validation_pass") is True, val.get("failure_reasons")
assert val.get("engine_replay_verified") is True, val
z=np.load(root/f"{stem}.npz", allow_pickle=True)
for k in ["trajectory","ball_trajectory","paddle_trajectory","state_time","state_qpos","state_qvel","state_mocap_pos","state_mocap_quat","metadata","model_xml","scene_description_plain","scene_description_precise","object_descriptions","mesh_manifest"]:
    assert k in z.files, k
assert len(list(root.glob(f"{stem}_*.mp4"))) == views
assert len(list(root.glob(f"{stem}_*_perception.json.gz"))) == views
assert len(list(root.glob(f"{stem}_*_raw_rgb_frames.tar.gz"))) == views
for p in root.glob(f"{stem}_*_perception.json.gz"):
    with gzip.open(p, "rt", encoding="utf-8") as f: d=json.load(f)
    assert d.get("frames"), p.name
    assert "objects" in d["frames"][0], p.name
mesh_objs=list((root/f"{stem}_meshes").glob("*.obj"))
assert len(mesh_objs)>=2, [p.name for p in mesh_objs]
print(json.dumps({"episode":eid,"validation_pass":True,"mp4":views,"raw_rgb_archives":views,"mesh_objs":len(mesh_objs)}, ensure_ascii=False))
PY
}

run_episode() {
  local ep=$1 batch_dir=$2 wid=$3 eid ep_dir seed attempt rc gpu
  eid=$(printf '%06d' "$ep"); ep_dir=$batch_dir/episode_$eid; seed=$((SEED0 + ep * 19))
  IFS=',' read -ra GPUS <<< "$GPU_LIST"; gpu=${GPUS[$((wid % ${#GPUS[@]}))]}
  rm -rf -- "$ep_dir"
  for attempt in 1 2 3; do
    require_tmp_space; mkdir -p "$ep_dir"; log "worker $wid episode $eid attempt $attempt start gpu=$gpu"
    set +e
    env TMPDIR="$TMP_ROOT/tmp" MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="$gpu" "$PY" "$SCRIPT" --episodes 1 --start-id "$ep" --seed "$seed" --out "$ep_dir" --videos 1 --video-cameras "$CAMERAS" --width "$WIDTH" --height "$HEIGHT" --copy-episode-meshes --keep-raw-rgb-frames --validation-retries 2 --progress-every 1 > "$ep_dir/run_attempt_${attempt}.log" 2>&1
    rc=$?; set -e; cleanup_tmp
    if [ "$rc" = 0 ] && validate_episode_dir "$ep_dir" "$ep" > "$ep_dir/strict_validation_summary.json" 2> "$ep_dir/strict_validation_error.log"; then
      log "worker $wid episode $eid ok strict state replay/bbox/mask/raw-rgb/mesh validated"; return 0
    fi
    log "worker $wid episode $eid attempt $attempt failed rc=$rc"
    tail -120 "$ep_dir/run_attempt_${attempt}.log" >> "$LOG_DIR/episode_${eid}_attempt_${attempt}_tail.log" 2>/dev/null || true
    cat "$ep_dir/strict_validation_error.log" >> "$LOG_DIR/episode_${eid}_attempt_${attempt}_tail.log" 2>/dev/null || true
    rm -rf -- "$ep_dir"; sleep $((attempt*15))
  done
  write_status failed "$ep" "worker_${wid}_episode_${eid}_failed"; return 1
}

archive_batch_locked() {
  local batch_dir=$1 start=$2 end=$3 expected=$4
  local got validations mp4 raw perception meshes desc tarball tmp_tar size checksum part_count
  got=$(find "$batch_dir" -type f -name 'episode_*.npz' | wc -l)
  validations=$(find "$batch_dir" -type f -name 'state_replay_validation.json' | wc -l)
  mp4=$(find "$batch_dir" -type f -name 'episode_*.mp4' | wc -l)
  raw=$(find "$batch_dir" -type f -name 'episode_*_raw_rgb_frames.tar.gz' | wc -l)
  perception=$(find "$batch_dir" -type f -name 'episode_*_perception.json.gz' | wc -l)
  meshes=$(find "$batch_dir" -type f -path '*_meshes/*.obj' | wc -l)
  desc=$(find "$batch_dir" -type f -name 'episode_*_descriptions.json' | wc -l)
  if [ "$got" -ne "$expected" ] || [ "$validations" -ne "$expected" ] || [ "$mp4" -ne $((expected*EXPECTED_VIEWS)) ] || [ "$raw" -ne $((expected*EXPECTED_VIEWS)) ] || [ "$perception" -ne $((expected*EXPECTED_VIEWS)) ] || [ "$meshes" -lt $((expected*2)) ] || [ "$desc" -ne "$expected" ]; then
    log "archive validation failed batch=$batch_dir got=$got validations=$validations mp4=$mp4 raw=$raw perception=$perception meshes=$meshes desc=$desc expected=$expected"; return 1
  fi
  tarball=$ARCHIVE_DIR/demo3_table_tennis_track_full_episodes_$(printf '%06d' "$start")_$(printf '%06d' "$end").tar.gz
  tmp_tar=$ARCHIVE_DIR/.tmp_$(basename "$tarball")
  rm -f -- "$tmp_tar" "$tarball" "$tarball.sha256" "$tarball".part* "$tarball.parts.sha256" "$tarball.parts.json"
  tar -C "$(dirname "$batch_dir")" -czf "$tmp_tar" "$(basename "$batch_dir")"
  tar -tzf "$tmp_tar" >/dev/null
  mv "$tmp_tar" "$tarball"
  size=$(stat -c %s "$tarball")
  checksum=$(sha256sum "$tarball" | awk '{print $1}')
  printf '%s  %s\n' "$checksum" "$(basename "$tarball")" > "$tarball.sha256"
  if (( size > 95000000 )); then
    split -b 90M -d -a 3 "$tarball" "$tarball.part"
    rm -f -- "$tarball"
    sha256sum "$tarball".part[0-9][0-9][0-9] > "$tarball.parts.sha256"
    part_count=$(find "$ARCHIVE_DIR" -maxdepth 1 -type f -name "$(basename "$tarball").part[0-9][0-9][0-9]" | wc -l)
    python3 - "$tarball" "$start" "$end" "$expected" "$size" "$checksum" "$part_count" <<'PY_ARCHIVE' >> "$META_DIR/archive_index.jsonl"
import json, sys, hashlib
from pathlib import Path
base=Path(sys.argv[1]); start=int(sys.argv[2]); end=int(sys.argv[3]); expected=int(sys.argv[4]); size=int(sys.argv[5]); checksum=sys.argv[6]; part_count=int(sys.argv[7])
parts=[]
for p in sorted(base.parent.glob(base.name + '.part[0-9][0-9][0-9]')):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    parts.append({'file': str(p), 'name': p.name, 'bytes': p.stat().st_size, 'sha256': h.hexdigest()})
(base.parent/(base.name+'.parts.json')).write_text(json.dumps(parts, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'archive': str(base), 'archive_sha256': checksum, 'split': True, 'part_count': part_count, 'parts': parts, 'start_episode': start, 'end_episode': end, 'episode_count': expected, 'bytes': size, 'state_replay_validated': True, 'contains_raw_rgb_frames': True, 'contains_2d_bboxes': True, 'contains_pixel_masks': True, 'contains_episode_mesh_assets': True}, ensure_ascii=False))
PY_ARCHIVE
    log "archived strict episodes $(printf '%06d' "$start")-$(printf '%06d' "$end") size=$size split_parts=$part_count"
  else
    python3 - "$tarball" "$start" "$end" "$expected" "$size" "$checksum" <<'PY_ARCHIVE_SINGLE' >> "$META_DIR/archive_index.jsonl"
import json, sys
print(json.dumps({'archive': sys.argv[1], 'sha256': sys.argv[6], 'split': False, 'start_episode': int(sys.argv[2]), 'end_episode': int(sys.argv[3]), 'episode_count': int(sys.argv[4]), 'bytes': int(sys.argv[5]), 'state_replay_validated': True, 'contains_raw_rgb_frames': True, 'contains_2d_bboxes': True, 'contains_pixel_masks': True, 'contains_episode_mesh_assets': True}, ensure_ascii=False))
PY_ARCHIVE_SINGLE
    log "archived strict episodes $(printf '%06d' "$start")-$(printf '%06d' "$end") size=$size"
  fi
  rm -rf -- "$batch_dir"; cleanup_tmp
}

archive_batch() {
  (
    flock 9
    archive_batch_locked "$@"
  ) 9>"$LOCK_DIR/archive.lock"
}

worker_loop() {
  local wid=$1 claim start end expected batch_dir ep rc
  log "worker $wid start"
  while claim=$(claim_batch); do
    start=$(echo "$claim" | awk '{print $1}'); end=$(echo "$claim" | awk '{print $2}')
    expected=$((end - start + 1))
    batch_dir=$TMP_ROOT/batches/worker_${wid}_batch_$(printf '%06d' "$start")_$(printf '%06d' "$end")
    rm -rf -- "$batch_dir"; mkdir -p "$batch_dir"
    write_worker_status "$wid" running "$start" "$end" batch_start
    write_status running "$start" "parallel_worker_${wid}_batch_$(printf '%06d' "$start")_$(printf '%06d' "$end")"
    for ((ep=start; ep<=end; ep++)); do run_episode "$ep" "$batch_dir" "$wid"; done
    archive_batch "$batch_dir" "$start" "$end" "$expected"
    write_worker_status "$wid" archived "$start" "$end" batch_archived
    write_status pushing "$end" "parallel_archived_$(printf '%06d' "$start")_$(printf '%06d' "$end")"
    if [ "$PUSH_EVERY_BATCH" = "1" ]; then sync_repo_and_push "parallel_episodes_$(printf '%06d' "$start")_$(printf '%06d' "$end")" || log "github push failed but local archive preserved for batch $start-$end"; fi
    write_worker_status "$wid" done "$start" "$end" batch_done
  done
  write_worker_status "$wid" complete "$TOTAL_EPISODES" "$TOTAL_EPISODES" no_more_claims
  log "worker $wid complete"
}

main() {
  [ -x "$PY" ] || { echo "missing python $PY" >&2; exit 2; }
  [ -f "$SCRIPT" ] || { echo "missing script $SCRIPT" >&2; exit 3; }
  [ -f "$VERIFY" ] || { echo "missing validator $VERIFY" >&2; exit 4; }
  write_repo_files
  if [ "${RESET_CLAIM:-0}" = "1" ]; then rm -f -- "$CLAIM_FILE"; fi
  if [ ! -f "$CLAIM_FILE" ]; then echo "$(first_missing_episode)" > "$CLAIM_FILE"; fi
  write_status running "$(cat "$CLAIM_FILE")" parallel_start
  sync_repo_and_push parallel_start || true
  log "parallel pipeline start total=$TOTAL_EPISODES workers=$WORKERS worker_batch_size=$WORKER_BATCH_SIZE claim_start=$(cat "$CLAIM_FILE") gpu_list=$GPU_LIST"
  local pids=() wid
  for ((wid=0; wid<WORKERS; wid++)); do worker_loop "$wid" >> "$LOG_DIR/worker_${wid}.log" 2>&1 & pids+=("$!"); done
  local fail=0
  for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  if [ "$fail" = 0 ]; then
    write_status complete "$TOTAL_EPISODES" parallel_complete
    sync_repo_and_push parallel_complete || true
    log "parallel pipeline complete"
  else
    write_status failed "$(cat "$CLAIM_FILE" 2>/dev/null || echo 0)" parallel_worker_failed
    log "parallel pipeline failed"
    exit 1
  fi
}
main "$@"
