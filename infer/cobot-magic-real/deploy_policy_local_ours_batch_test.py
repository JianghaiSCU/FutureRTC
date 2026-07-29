"""ours 真机**正式批量测试**脚本（独立，不改动 deploy_policy_local_ours_local.py）。

与 deploy_policy_local_ours_local.py 的控制逻辑完全一致（同一个 S=25 / d=10 /
Cn[0:S] 契约、同一套插值与异步查询），只是外面套了一层正式实验的批量测试流程：

流程（每个任务 --episodes 轮，默认 20）:
  1. 启动 → 硬件 init → 机械臂回 0 位。
  2. 每一轮:
     a. 打印提示，**等你放好道具后回车**开始。
     b. 跑 ours 控制循环（计时从机械臂开始动那一刻起）。
        结束条件二选一:
          - 你**回车**（人为判定任务结束 / 失败）→ **立即停**（当前控制点之后）;
          - 累计**已执行的 raw action 数 >= --max-step**（默认 300）→ 自动停。
     c. 让你输入**成功 / 失败**（s/f 或 1/0 或 y/n）。
     d. **立即把这一轮的结果 append 到 results.jsonl 并 flush**（中途崩了也不丢前面）。
     e. 机械臂**平滑插值回 0 位**，回到 (a) 等下一轮。
  3. 20 轮跑完，计算并写出**总成功率 SR、平均执行步数、平均执行时间**。

指标（每轮）:
  - success            : 人为判定
  - executed_steps     : **模型输出的 action 里被真正执行的 raw action 数**
                         （不是 H=50 全部输出，也不是插值后的控制点数）
  - exec_time_s        : 机械臂开始动到这一轮结束的墙钟时间
  - obs2act_ms (中间量): 每次查询 obs 发出 → 返回 action 的 HTTP 往返时间
                         （**不含** MANUAL_DELAY=0.15 那段人为 sleep，量的是真实
                          服务端推理往返），记录每轮的 mean / max

**policy server 开了就不停不重载**（模型只加载一次）。但每轮开头仍会发一个
``flag='reset'``——这只是清 server 端 ours 特有的 **z_init 缓存**（每 episode 首帧
的 latent 参考），是 ours 正确性必需的（不发的话第 2~20 轮会拿第 1 轮的首帧当参考），
**不是**重启 server / 重载模型，很轻。

用法:
  source set_player.sh 1
  python deploy_policy_local_ours_batch_test.py --task plates_stacking --episodes 20 --max-step 300
"""

from control_joints import *
from read_joints import *
from realsense import *

import numpy as np
from pathlib import Path
import sys
import os
import time
import json
import queue
import shutil
import subprocess
import threading
import argparse

sys.path.append(str(Path(__file__).resolve().parent))

import pickle
import base64
import requests
import cv2

current_file_path = os.path.abspath(__file__)


def _find_repo_root(start: Path) -> Path:
    for cand in [start, *start.parents]:
        if (cand / "ours_pi05" / "__init__.py").is_file():
            return cand
    raise RuntimeError(
        f"往上找不到 ours_pi05/ 包（从 {start} 起）。请把 ours_pi05/ 放在 cobot-magic-real/ 的同级目录。"
    )


_REPO_ROOT = _find_repo_root(Path(current_file_path).resolve().parent)
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ours_pi05.deploy_protocol import (  # noqa: E402
    DELAY_STEPS,
    FIRE_STEP,
    INTERP_STEP_SIZES,
    METHOD,
    S,
    committed_actions_from_slice,
    executed_slice,
)

CONTROL_FREQUENCY = 30.0  # 与 deploy_policy_local_ours_local.py 一致
MANUAL_DELAY = 0.15       # 与原脚本一致（后台查询里 POST 之前的人为 sleep）


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ours 真机正式批量测试")
    p.add_argument("--task", required=True, help="任务名，用于结果文件名")
    p.add_argument("--server-url", default="http://127.0.0.1:8001/send")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--max-step", type=int, default=300,
                   help="一轮最多执行多少个 raw action（模型输出被执行的动作数），到了自动结束")
    p.add_argument("--results-dir", default="outputs/batch_test",
                   help="结果根目录，实际写到 <results-dir>/<task>/（每任务固定目录，便于断点续跑）")
    p.add_argument("--save-video", action="store_true",
                   help="保存执行期间的 RealSense 视频（left|front|right 三视角横向拼接，mp4）。"
                        "执行期间只抓帧存内存，mp4 编码在本轮结束后的空闲段做，不影响计时。")
    p.add_argument("--video-fps", type=int, default=15, help="录制帧率（默认 15，够看且省内存/磁盘）")
    p.add_argument("--no-save-actions", dest="save_actions", action="store_false",
                   help="不保存每轮被执行的 raw action（默认保存）")
    p.add_argument("--fresh", action="store_true",
                   help="强制重新开始一整轮（把已存在的同任务结果目录改名归档，从第 1 轮起）")
    p.add_argument("--instruction-file", default="instruction.txt")
    return p.parse_args()


# ---------------------------------------------------------------------------
# obs 编码 / 插值（与 deploy_policy_local_ours_local.py 逐字一致）
# ---------------------------------------------------------------------------
def encode_pickle_base64(obj):
    return base64.b64encode(pickle.dumps(obj)).decode('utf-8')


def encode_obs(observation):
    obs = {
        'input_rgb_arr': [
            observation['head_camera'].copy(),
            observation['left_camera'].copy(),
            observation['right_camera'].copy(),
        ],
        'input_state': observation['joint_action'],
    }
    return obs


def interpolate_action_sequence_with_boundaries(actions):
    """与原脚本一致。boundaries[k] = 到并含第 k 个原始动作为止的插值点数
    （即原始动作 k 结束于 smoothed 索引 boundaries[k]-1）。boundaries[0]=1 是起点，
    boundaries[1..T-1] 对应 actions[1..T-1]。"""
    steps = np.asarray(INTERP_STEP_SIZES)
    smoothed_actions = [actions[0][np.newaxis, :]]
    boundaries = [1]
    count = 1
    T = actions.shape[0]
    for i in range(T - 1):
        prev_action = actions[i]
        cur_action = actions[i + 1]
        diff = np.abs(cur_action - prev_action)
        step = np.ceil(diff / steps).astype(int)
        max_step = int(np.max(step))
        if max_step <= 1:
            seg = cur_action[np.newaxis, :]
        else:
            seg = np.linspace(prev_action, cur_action, max_step + 1)[1:]
        smoothed_actions.append(seg)
        count += seg.shape[0]
        boundaries.append(count)
    return np.concatenate(smoothed_actions, axis=0), boundaries


def clip_grippers(position):
    if position[6] <= 0.01:
        position[6] = 0
    if position[13] <= 0.01:
        position[13] = 0
    return position


class VideoRecorder:
    """独立线程录制执行期间的三视角拼接视频（left | front(head) | right）。

    **为了不影响时序指标：执行期间只做「抓帧 + 拼接 -> 存内存」，不编码。**
    mp4 编码（重活）在 ``flush()`` 里做，调用时机是这一轮**执行结束之后**的空闲段
    （输入成功/失败 + 回 0 位那会儿），彻底挪出计时窗口。执行期间录像线程占 GIL 的
    只剩两次 numpy 拷贝、约 1ms/帧，且默认 15fps（唤醒频率减半），对 30Hz 控制循环
    的抖动可忽略。相机自身有后台线程更新最新帧，``get_latest_image()`` 只读不阻塞、
    不碰 CAN reader。帧是 RGB 640x480 拼成 1920x480，编码时才转 BGR。

    内存：一帧 1920x480x3 ≈ 2.76MB，15fps 下一轮 ~30s 约 1.2GB，机器内存足够。
    """

    def __init__(self, cams_in_order, path, fps=15):
        self.cams = cams_in_order  # 必须是 [left, head(front), right] 顺序
        self.path = str(path)
        self.fps = int(fps)
        self._stop = threading.Event()
        self._thread = None
        self._frames = []          # 执行期间攒下的 RGB 拼接帧
        self.n_frames = 0

    def _grab_stitched(self):
        imgs = []
        for cam in self.cams:
            im = cam.get_latest_image()
            if im is None:
                return None
            imgs.append(np.ascontiguousarray(im))
        return np.hstack(imgs)      # (480, 1920, 3) RGB，先不转 BGR（省一次拷贝）

    def _loop(self):
        period = 1.0 / self.fps
        last = None
        while not self._stop.is_set():
            t = time.time()
            frame = self._grab_stitched()
            if frame is not None:
                self._frames.append(frame)
                last = frame
            elif last is not None:
                self._frames.append(last)   # 某相机瞬时无帧，复用上一帧保持时长
            dt = period - (time.time() - t)
            if dt > 0:
                time.sleep(dt)

    def start(self):
        self._stop.clear()
        self._frames = []
        self.n_frames = 0
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """停止抓帧（执行结束时调，很快返回，不编码）。"""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.n_frames = len(self._frames)

    def flush(self):
        """把攒下的帧编码成 mp4。**在计时窗口之外调**（本轮结束后的空闲段）。

        优先用 ffmpeg 出 **H.264（libx264）** —— 这样 VSCode / 浏览器能直接播；
        pip 装的 OpenCV 不带 H.264 编码器（licensing），cv2 只能出 mp4v，很多播放器
        不认。没 ffmpeg 才回退 cv2 mp4v。
        """
        if not self._frames:
            return
        h, w = self._frames[0].shape[:2]
        if not self._encode_ffmpeg_h264(w, h):
            self._encode_cv2_mp4v(w, h)   # 兜底
        self._frames = []                 # 释放内存

    def _encode_ffmpeg_h264(self, w, h) -> bool:
        """把 RGB 帧管道喂给 ffmpeg 编码 H.264。成功返回 True。"""
        if shutil.which("ffmpeg") is None:
            return False
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", str(self.fps), "-i", "-",
            "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", self.path,
        ]
        try:
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
            for rgb in self._frames:
                p.stdin.write(np.ascontiguousarray(rgb).tobytes())  # RGB24
            p.stdin.close()
            return p.wait() == 0
        except Exception as e:  # noqa: BLE001
            print(f"[VIDEO] ffmpeg 编码失败，回退 cv2 mp4v：{e}")
            return False

    def _encode_cv2_mp4v(self, w, h) -> None:
        writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        for rgb in self._frames:
            writer.write(np.ascontiguousarray(rgb[:, :, ::-1]))  # RGB -> BGR
        writer.release()


# ---------------------------------------------------------------------------
# 硬件 / server 设置（与 deploy_policy_local_ours_local.py 一致）
# ---------------------------------------------------------------------------
args = parse_args()

player_value = os.getenv("PLAYER")
print(f"[INIT] PLAYER: {player_value}")
if player_value is None:
    raise ValueError("环境变量 PLAYER 未设置")
try:
    player_value = int(player_value)
except ValueError:
    raise ValueError("环境变量 PLAYER 必须是一个整数")

if player_value == 1:
    cameras = [
        RealSenseCam("244222071389", "left_camera"),
        RealSenseCam("242222071721", "head_camera"),
        RealSenseCam("313522070980", "right_camera"),
    ]
    left_can, right_can = "can_left_1", "can_right_1"
elif player_value == 2:
    cameras = [
        RealSenseCam("244222071389", "left_camera"),
        RealSenseCam("242222071721", "head_camera"),
        RealSenseCam("313522070980", "right_camera"),
    ]
    left_can, right_can = "can_left_2", "can_right_2"
else:
    raise ValueError("PLAYER 值无效，必须是 1 或 2")

print(f"[INIT] left_can={left_can}  right_can={right_can}")

ctx = rs.context()
if len(ctx.devices) > 0:
    print("[INIT] RealSense devices:")
    for d in ctx.devices:
        print(f"  {d.get_info(rs.camera_info.name)}  SN={d.get_info(rs.camera_info.serial_number)}")
else:
    print("[INIT] No RealSense devices connected")

for cam in cameras:
    cam.start()
print("[INIT] Warming up cameras...")
for i in range(20):
    print(f'[INIT] Warm up: {i}/20', end='\r')
    for cam in cameras:
        _ = cam.get_latest_image()
    time.sleep(0.1)
print("\n[INIT] Camera warm up done!")

reader = JointReader(left_can=left_can, right_can=right_can)
controller = ControlJoints(left_can=left_can, right_can=right_can)
print("[INFO] Real Controller initialized. Robot will MOVE.")

with open(args.instruction_file, 'r', encoding='utf-8') as f:
    instruction = f.readline().strip()
print(f"[INIT] Instruction: {instruction}")

SERVER_URL = args.server_url
print(f"[INIT] Server URL: {SERVER_URL}")


# ---------------------------------------------------------------------------
# obs 捕获 + 查询（query_server 记录 obs->action 往返时间）
# ---------------------------------------------------------------------------
def capture_and_encode():
    data = dict()
    for _ in range(3):
        for cam in cameras:
            color_image = cam.get_latest_image()
            if color_image is not None:
                data[cam.name] = color_image
    joint_values = reader.get_joint_value()
    data['joint_action'] = joint_values
    return encode_obs(data), np.asarray(joint_values, dtype=np.float32)


def send_reset():
    """清 server 端 z_init 缓存（ours 每 episode 首帧参考）。见文件头说明。"""
    r = requests.post(SERVER_URL, json={'data': encode_pickle_base64({}), 'flag': 'reset'}, timeout=50)
    r.raise_for_status()


# 每轮的 obs->action 往返时间（ms）。query_server 在此追加（list.append 在 CPython
# 下受 GIL 保护，主线程 + 后台 fetcher 线程并发 append 安全）。每轮开头清空。
latency_ms: list = []


def query_server(obs, committed_actions=None, inference_delay=0):
    committed = np.zeros((0, 14), dtype=np.float32) if committed_actions is None \
        else np.asarray(committed_actions, dtype=np.float32)
    payload = {
        'data': encode_pickle_base64(obs),
        'flag': 'return',
        'method': METHOD,
        'inference_delay': int(inference_delay),
        'committed_actions': committed.tolist(),
    }
    t0 = time.time()
    response = requests.post(SERVER_URL, json=payload, timeout=50)
    dt_ms = (time.time() - t0) * 1000.0
    response.raise_for_status()
    latency_ms.append(dt_ms)  # obs 发出 -> action 返回（不含 MANUAL_DELAY）
    return np.array(response.json()["result"])


class AsyncChunkFetcher:
    def __init__(self, url, manual_delay=0.0):
        self.url = url
        self.manual_delay = manual_delay
        self._done = threading.Event()
        self._result = None
        self._err = None
        self._done.set()

    def submit(self, obs, committed_actions=None, inference_delay=0):
        self._done.clear()
        self._result = None
        self._err = None
        threading.Thread(target=self._worker,
                         args=(obs, committed_actions, inference_delay), daemon=True).start()

    def _worker(self, obs, committed_actions, inference_delay):
        try:
            if self.manual_delay > 0:
                time.sleep(self.manual_delay)
            self._result = query_server(obs, committed_actions=committed_actions,
                                        inference_delay=inference_delay)
        except Exception as e:  # noqa: BLE001
            self._err = e
        finally:
            self._done.set()

    def get(self, timeout=None):
        if not self._done.wait(timeout):
            raise TimeoutError("async chunk fetch timed out")
        if self._err is not None:
            raise self._err
        return self._result


# ---------------------------------------------------------------------------
# stdin 单一读取线程 -> 队列。主线程在不同时机消费:
#   - 等你回车开始 / 输入成功失败: 阻塞取;
#   - 执行中: 非阻塞 poll，收到回车即立即停。
# 只用一个线程读 stdin，避免多个 input() 抢占。
# ---------------------------------------------------------------------------
_stdin_q: queue.Queue = queue.Queue()


def _stdin_reader():
    for line in sys.stdin:
        _stdin_q.put(line.rstrip("\n"))


threading.Thread(target=_stdin_reader, daemon=True).start()


def drain_stdin():
    while not _stdin_q.empty():
        try:
            _stdin_q.get_nowait()
        except queue.Empty:
            break


def wait_line(prompt=None):
    if prompt is not None:
        print(prompt, flush=True)
    return _stdin_q.get()


def poll_line():
    try:
        return _stdin_q.get_nowait()
    except queue.Empty:
        return None


def ask_keep() -> bool:
    """这一轮算不算数。道具没放好、中途碰到、明显的意外 -> 丢弃重跑，不计入统计。"""
    while True:
        ans = wait_line("[ASK] 这一轮保留还是丢弃? 保留=k/1/y(直接回车也是保留)  丢弃=d/0/n : ").strip().lower()
        if ans in ("", "k", "1", "y", "keep", "保留"):
            return True
        if ans in ("d", "0", "n", "drop", "discard", "丢弃"):
            return False
        print("[ASK] 没听懂，重新输入。")


def ask_success() -> bool:
    while True:
        ans = wait_line("[ASK] 这一轮成功还是失败? 成功=s/1/y  失败=f/0/n : ").strip().lower()
        if ans in ("s", "1", "y", "success", "成功"):
            return True
        if ans in ("f", "0", "n", "fail", "失败"):
            return False
        print("[ASK] 没听懂，重新输入。")


GRIPPER_OPEN = 0.1   # 抓夹物理范围 0~0.1：0.1=张开，0=闭合（参考 reset.py）


def move_to_zero():
    """回 0 位：臂平滑插值回全 0，**同时抓夹张开松手**；到位后停顿让物体落下，再闭合抓夹。

    不加张开这一步的话，上一轮抓着的东西会一直攥着不放（参考 reset.py 的
    张开(0.1)->闭合(0)）。"""
    cur = np.asarray(reader.get_joint_value(), dtype=np.float32)
    # 目标：臂全 0，抓夹张开(0.1) —— 插值过程中抓夹逐步张开松手
    target = np.zeros(14, dtype=np.float32)
    target[6] = target[13] = GRIPPER_OPEN
    smoothed, _ = interpolate_action_sequence_with_boundaries(np.stack([cur, target]))
    for position in smoothed:
        # 这一段**不** clip 抓夹：要的就是张开，clip 会把小值吸到 0(闭合) 反而打架
        t = time.time()
        controller.control(position)
        dt = 1.0 / CONTROL_FREQUENCY - (time.time() - t)
        if dt > 0:
            time.sleep(dt)
    # 确保臂到位 + 抓夹完全张开，再停顿让抓着的东西落下
    open_pose = [0.0] * 14
    open_pose[6] = open_pose[13] = GRIPPER_OPEN
    for _ in range(3):
        controller.control(open_pose)
        time.sleep(0.05)
    time.sleep(0.3)  # 松手停顿
    # 抓夹闭合，回到全 0 初始位
    for _ in range(3):
        controller.control([0] * 14)
        time.sleep(0.05)


# ---------------------------------------------------------------------------
# 跑一轮 ours 控制循环。返回 (executed_steps, exec_time_s, stop_reason, n_queries)
# ---------------------------------------------------------------------------
def run_one_episode(video_path=None) -> dict:
    global latency_ms
    latency_ms = []
    last_position = None
    window_idx = 0
    executed_steps = 0            # 本轮累计已执行的 raw action 数
    executed_actions = []       # 每 window **真正执行掉**的 raw action（非插值控制点）
    executed_window_ids = []    # 与之对齐的 window 序号，便于事后按 chunk 边界切
    stop_reason = None
    fetcher = AsyncChunkFetcher(SERVER_URL, manual_delay=MANUAL_DELAY)

    # 清 z_init（见文件头）
    send_reset()

    # bootstrap: 阻塞查 C0（delay=0，无 committed；server 此时也缓存 z_init）
    obs_enc, start_joints = capture_and_encode()
    current_chunk = query_server(obs_enc, committed_actions=None, inference_delay=0)

    drain_stdin()  # 丢掉「回车开始」可能残留的换行，避免一进来就立即停

    # 只在执行期间录（bootstrap 不录）。cameras 顺序就是 [left, head, right]。
    recorder = None
    if video_path is not None:
        recorder = VideoRecorder(cameras, video_path, fps=args.video_fps)
        recorder.start()

    t_exec_start = time.time()    # 机械臂即将开始动 —— 执行计时从这里起

    while stop_reason is None:
        try:
            action_slice = executed_slice(current_chunk, S)  # Cn[0:S]
        except ValueError:
            stop_reason = "short_chunk"
            break

        committed = committed_actions_from_slice(action_slice, fire_step=FIRE_STEP, delay=DELAY_STEPS)

        start = start_joints if last_position is None else last_position
        positions = np.concatenate(([start], action_slice), axis=0)  # (S+1, 14)
        smoothed, boundaries = interpolate_action_sequence_with_boundaries(positions)
        fire_at = boundaries[FIRE_STEP] - 1

        fired = False
        raw_ptr = 1            # 下一个待完成的 raw action 边界（boundaries[0] 是起点）
        raw_done_window = 0    # 本 window 已完成的 raw action 数
        stop_now = False

        for j, position in enumerate(smoothed):
            clip_grippers(position)
            t = time.time()
            controller.control(position)
            dt = 1.0 / CONTROL_FREQUENCY - (time.time() - t)
            if dt > 0:
                time.sleep(dt)
            last_position = position

            # 数已完成的 raw action（可能一个控制点跨过多个短段边界）
            while raw_ptr < len(boundaries) and boundaries[raw_ptr] - 1 <= j:
                raw_done_window += 1
                raw_ptr += 1

            # 中途发下一次查询（只有不停时才发；停了就不需要下一块）
            if (not fired) and (j >= fire_at):
                obs_next, _ = capture_and_encode()
                fetcher.submit(obs_next, committed_actions=committed, inference_delay=DELAY_STEPS)
                fired = True

            # 立即停检查（当前控制点之后）: 人为回车 或 达到 max_step
            if poll_line() is not None:
                stop_reason = "human"
                stop_now = True
                break
            if executed_steps + raw_done_window >= args.max_step:
                stop_reason = "max_step"
                stop_now = True
                break

        executed_steps += raw_done_window

        # 记下这一 window **真正被执行掉**的 raw action。中途停的话只有前
        # raw_done_window 个真的执行了，action_slice 剩下的没下发过，不能算。
        if raw_done_window > 0:
            executed_actions.append(np.asarray(action_slice[:raw_done_window], dtype=np.float32))
            executed_window_ids.append(np.full(raw_done_window, window_idx, dtype=np.int32))

        if stop_now:
            break

        # window 走完（没停）: 取下一块继续
        if not fired:
            # window 短于 fire 点（S=25/d=10 不会发生）: 在边界补发
            obs_next, _ = capture_and_encode()
            fetcher.submit(obs_next, committed_actions=committed, inference_delay=DELAY_STEPS)
        try:
            current_chunk = fetcher.get(timeout=50)
        except Exception as e:
            print(f"[ASYNC] next chunk fetch failed: {e}")
            stop_reason = "fetch_error"
            break
        window_idx += 1

    exec_time_s = time.time() - t_exec_start
    # 只停抓帧（很快返回，不编码）。mp4 编码由 main() 在计时窗口之外调 recorder.flush()。
    if recorder is not None:
        recorder.stop()
    acts = (np.concatenate(executed_actions, axis=0) if executed_actions
            else np.zeros((0, 14), dtype=np.float32))
    wids = (np.concatenate(executed_window_ids, axis=0) if executed_window_ids
            else np.zeros((0,), dtype=np.int32))
    if acts.shape[0] != executed_steps:
        # 构造上二者恒等；不等说明计步或切片有 bug，别让它静默过去
        print(f"[WARN] 存下的 action 数 {acts.shape[0]} != executed_steps {executed_steps}，"
              f"两者本应恒等。结果里的 n_actions_saved 会留下证据。")
    lat = list(latency_ms)
    res = {
        "executed_steps": int(executed_steps),
        "exec_time_s": float(exec_time_s),
        "stop_reason": stop_reason,
        "n_queries": len(lat),
        "obs2act_ms_mean": float(np.mean(lat)) if lat else 0.0,
        "obs2act_ms_max": float(np.max(lat)) if lat else 0.0,
        "n_actions_saved": int(acts.shape[0]),
        "windows": int(window_idx),
        "video": (str(video_path) if video_path is not None else None),
        "video_frames": (recorder.n_frames if recorder is not None else 0),
    }
    return res, recorder, acts, wids


# ---------------------------------------------------------------------------
# 批量测试主流程
# ---------------------------------------------------------------------------
def main():
    # 每任务固定目录（不带时间戳）→ 断点续跑：再次运行同一 --task 会自动接着记录。
    out_dir = Path(args.results_dir) / args.task
    jsonl_path = out_dir / "results.jsonl"    # 每轮 append + flush，崩溃安全
    txt_path = out_dir / "results.txt"        # 人类可读的滚动表
    actions_dir = out_dir / "actions"
    video_dir = out_dir / "video"

    # --fresh：把已存在的旧结果目录改名归档，从头开始。
    if args.fresh and out_dir.exists():
        archived = out_dir.with_name(f"{args.task}_archived_{time.strftime('%Y%m%d_%H%M%S')}")
        out_dir.rename(archived)
        print(f"[BATCH] --fresh: 旧结果已归档到 {archived}")

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_actions:
        actions_dir.mkdir(parents=True, exist_ok=True)
    if args.save_video:
        video_dir.mkdir(parents=True, exist_ok=True)

    # 断点续跑：读已完成的轮，从下一轮起。
    records = []
    if jsonl_path.exists():
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    completed = len(records)
    start_ep = completed + 1

    print(f"\n[BATCH] task={args.task}  目标 {args.episodes} 轮  max_step={args.max_step}  "
          f"save_video={args.save_video}")
    print(f"[BATCH] 结果目录: {out_dir}")
    if completed:
        print(f"[BATCH] **检测到已完成 {completed} 轮，从第 {start_ep} 轮继续**（追加到原记录）")
    print(f"[BATCH] 每一轮结束立即写入 {jsonl_path.name} 并 fsync（中途中断也不丢前面）\n")

    # 表头只在文件新建时写，续跑不覆盖。
    if not txt_path.exists() or txt_path.stat().st_size == 0:
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"# ours batch test | task={args.task} | 目标 {args.episodes} 轮 "
                    f"| max_step={args.max_step}\n")
            f.write(f"{'ep':>3} {'success':>8} {'steps':>6} {'time_s':>8} "
                    f"{'stop':>11} {'obs2act_ms(mean/max)':>22}\n")

    if start_ep > args.episodes:
        print(f"[BATCH] 已完成 {completed} 轮 >= 目标 {args.episodes}，直接汇总。用 --fresh 重开一轮。")

    # 开测前先回 0 位
    if start_ep <= args.episodes:
        print("[BATCH] 机械臂回 0 位 ...")
        move_to_zero()

    # 轮次由**保留**的次数驱动：丢弃的那次不占编号、不计入，原地重跑。
    ep = start_ep
    n_discarded = 0
    while ep <= args.episodes:
        print(f"\n{'='*56}")
        print(f"[EPISODE {ep}/{args.episodes}] 放好任务道具后，按【回车】开始 ..."
              + (f"（累计丢弃 {n_discarded} 次）" if n_discarded else ""))
        drain_stdin()
        wait_line()  # 阻塞等你回车

        video_path = (video_dir / f"episode_{ep}.mp4") if args.save_video else None
        print(f"[EPISODE {ep}] 开始执行。人为结束按【回车】；或到 {args.max_step} 步自动停。"
              + (f" 录像 -> {video_path.name}" if video_path else ""))
        res, recorder, acts, wids = run_one_episode(video_path=video_path)  # 计时已结束，recorder 已停抓帧

        drain_stdin()  # 清掉停止用的那次回车/多按的键 —— ask_keep 把空行当"保留"，不清会被误判
        if not ask_keep():
            n_discarded += 1
            print(f"[EPISODE {ep}] **丢弃** —— 不计入，这一轮重跑（累计丢弃 {n_discarded} 次）。")
            recorder = None  # 扔掉帧缓存，不编码（video_path 留给重跑那次覆写）
            print(f"[EPISODE {ep}] 机械臂回 0 位 ...")
            move_to_zero()
            continue  # ep 不前进

        success = ask_success()

        rec = {"episode": ep, "success": success, **res, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        records.append(rec)

        # 立即落盘（append + flush）—— 关键，防中途断电/崩溃丢数据
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

        # 被执行的 raw action 落盘（**不是**插值后的控制点）。actions[i] 是第 i 个
        # 被执行的模型输出动作，window_ids[i] 是它属于哪个 window（chunk 边界）。
        if args.save_actions:
            np.savez(actions_dir / f"episode_{ep}.npz", actions=acts, window_ids=wids)
            print(f"[EPISODE {ep}] 已存 {acts.shape[0]} 个被执行的 raw action -> "
                  f"actions/episode_{ep}.npz")

        # mp4 编码在此做 —— **计时窗口之外**（已过 exec_time 计时、已记录指标），
        # 所以不影响 exec_time / obs->action。放在回 0 位之前的空闲段。
        if recorder is not None:
            print(f"[EPISODE {ep}] 编码录像（{recorder.n_frames} 帧）...")
            recorder.flush()
        with open(txt_path, "a", encoding="utf-8") as f:
            f.write(f"{ep:>3} {('成功' if success else '失败'):>8} "
                    f"{res['executed_steps']:>6} {res['exec_time_s']:>8.2f} "
                    f"{str(res['stop_reason']):>11} "
                    f"{res['obs2act_ms_mean']:>10.1f}/{res['obs2act_ms_max']:<8.1f}\n")
            f.flush()

        print(f"[EPISODE {ep}] {'成功' if success else '失败'} | "
              f"steps={res['executed_steps']} time={res['exec_time_s']:.2f}s "
              f"stop={res['stop_reason']} obs2act={res['obs2act_ms_mean']:.0f}/{res['obs2act_ms_max']:.0f}ms "
              f"-> 已写入 {jsonl_path.name}")

        # 回 0 位，等下一轮
        print(f"[EPISODE {ep}] 机械臂回 0 位 ...")
        move_to_zero()
        ep += 1

    if n_discarded:
        print(f"\n[BATCH] 本次运行共丢弃 {n_discarded} 轮（未计入统计）。")

    # ---- 汇总 ----
    n = len(records)
    n_succ = sum(r["success"] for r in records)
    sr = n_succ / n if n else 0.0
    steps_all = [r["executed_steps"] for r in records]
    time_all = [r["exec_time_s"] for r in records]
    succ = [r for r in records if r["success"]]
    steps_succ = [r["executed_steps"] for r in succ]
    time_succ = [r["exec_time_s"] for r in succ]
    lat_all = [r["obs2act_ms_mean"] for r in records]

    def _mean(xs):
        return float(np.mean(xs)) if xs else 0.0

    summary = {
        "task": args.task,
        "episodes": n,
        "success_count": n_succ,
        "success_rate": sr,
        "max_step": args.max_step,
        "mean_executed_steps_all": _mean(steps_all),
        "mean_exec_time_s_all": _mean(time_all),
        "mean_executed_steps_success": _mean(steps_succ),
        "mean_exec_time_s_success": _mean(time_succ),
        "mean_obs2act_ms": _mean(lat_all),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(f"\n{'='*56}")
    print(f"[SUMMARY] task={args.task}  {n} 轮")
    print(f"  成功率 SR       : {n_succ}/{n} = {sr*100:.1f}%")
    print(f"  平均执行步数     : 全部 {_mean(steps_all):.1f}   成功轮 {_mean(steps_succ):.1f}")
    print(f"  平均执行时间(s)  : 全部 {_mean(time_all):.2f}   成功轮 {_mean(time_succ):.2f}")
    print(f"  平均 obs->action : {_mean(lat_all):.0f} ms")
    print(f"  结果目录         : {out_dir}")
    print(f"{'='*56}")

    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(f"\n# SUMMARY  SR={n_succ}/{n}={sr*100:.1f}%  "
                f"steps(all/succ)={_mean(steps_all):.1f}/{_mean(steps_succ):.1f}  "
                f"time(all/succ)={_mean(time_all):.2f}/{_mean(time_succ):.2f}s  "
                f"obs2act={_mean(lat_all):.0f}ms\n")


def _cleanup():
    """关掉硬件后台进程/线程，否则脚本跑完不退出（要 Ctrl-C）。

    JointReader 起了两个 multiprocessing.Process 读 CAN —— **它们不是 daemon 线程，
    主线程结束后仍存活，会挂住 Python 主进程的退出**。这才是「跑完汇总后终端还在跑」
    的根因。相机线程是 daemon（不挡退出），但顺手也停掉释放 pipeline。
    """
    try:
        reader.close()   # terminate + join 两个 CAN 子进程（read_joints.py:87）
    except Exception as e:
        print(f"[CLEANUP] reader.close 出错（忽略）: {e}")
    for cam in cameras:
        try:
            cam.stop()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    finally:
        _cleanup()
        # 兜底：即使还有别的非 daemon 残留（如 SDK 内部线程），也确保进程退出，
        # 不让终端挂住。os._exit 跳过 atexit，此时结果已全部落盘。
        os._exit(0)
