import pyrealsense2 as rs
import numpy as np
import threading
from collections import deque
import cv2
import time
import os
import queue # 引入队列用于异步解耦

class RealSenseCam:
    def __init__(self, serial_number, name,depth_use=False):
        self.serial_number = serial_number
        self.name = name
        self.depth_use = depth_use
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(serial_number)
        # 只启用彩色图像流
        self.fps = 30
        self.width = 640
        self.height = 480
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        if self.depth_use:
            self.config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            self.align = rs.align(rs.stream.color)
            self.frame_buffer_depth = deque(maxlen=1)  # 仅保留最新一帧
        # 使用双端队列替换队列，简化帧管理
        self.frame_buffer = deque(maxlen=1)  # 仅保留最新一帧
        self.thread = None
        self.exit_event = threading.Event()
        # --- 优先级优化：录制专用队列和线程 ---
        self.is_recording = False
        self.record_queue = queue.Queue(maxsize=60) # 最多缓冲2秒(60帧)，防止撑爆内存
        self.record_thread = None

    def start(self):
        self.exit_event.clear()
        self.pipeline.start(self.config)
        self.thread = threading.Thread(target=self._update_frames, name=f"{self.name}_stream")
        self.thread.daemon = True  # 设置为守护线程
        self.thread.start()

    def _record_worker(self, filename):
        """独立的录制线程：负责耗时的磁盘 I/O"""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, self.fps, (self.width, self.height))
        
        print(f"Recording worker started: {filename}")
        while self.is_recording or not self.record_queue.empty():
            try:
                # 等待新帧，超时1秒
                frame = self.record_queue.get(timeout=1.0)
                out.write(frame)
                self.record_queue.task_done()
            except queue.Empty:
                continue
        
        out.release()
        print(f"Recording worker stopped: {filename}")

    def start_recording(self, save_path="recordings"):
        if self.is_recording: return
        
        if not os.path.exists(save_path): os.makedirs(save_path)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(save_path, f"{self.name}_{timestamp}.mp4")
        
        self.is_recording = True
        self.record_thread = threading.Thread(target=self._record_worker, args=(filename,))
        self.record_thread.start()

    def stop_recording(self):
        self.is_recording = False
        if self.record_thread and self.record_thread.is_alive():
            # 2. 等待录制线程把 record_queue 里剩下的帧写完并关闭文件
            # 这一步非常重要！没有这一步，Ctrl+C 可能会导致视频损坏
            self.record_thread.join(timeout=5) 
            self.record_thread = None
        print(f"[{self.name}] 录制已安全停止并保存。")

    def _update_frames(self):
        try:
            while not self.exit_event.is_set():
                # 等待彩色帧数据（超时5秒）
                frames = self.pipeline.wait_for_frames(5000)

                if self.depth_use:
                    frames = self.align.process(frames)
                    depth_frame = frames.get_depth_frame()
                    if depth_frame:
                        # 转换为NumPy数组并存储
                        depth_image = np.asanyarray(depth_frame.get_data())
                        self.frame_buffer_depth.append(depth_image)  # 保留最新帧    
                color_frame = frames.get_color_frame() 

                if color_frame:
                    # 转换为NumPy数组并存储
                    bgr_image = np.asanyarray(color_frame.get_data())
                    color_image = bgr_image[:, :, ::-1]
                    self.frame_buffer.append(color_image)  # 保留最新帧
                    # 优先步骤 2: 异步录制 (BGR)
                    if self.is_recording:
                        try:
                            # block=False 保证如果队列满了，立即跳过，绝不卡主线程
                            self.record_queue.put_nowait(bgr_image.copy())
                        except queue.Full:
                            pass # 磁盘太慢时丢弃录制帧，保证控制实时性                    
                          
        except Exception as e:
            print(f"Error from {self.name} camera: {e}")
        finally:
            self.stop_recording() # 确保退出时关闭文件
            self.pipeline.stop()

    def get_latest_image(self):
        if self.frame_buffer:
            return self.frame_buffer[-1]  # 返回最新一帧
        return None
    
    def get_latest_image_depth(self):
        if self.frame_buffer_depth:
            return self.frame_buffer_depth[-1]  # 返回最新一帧
        return None        

    # def stop(self):
    #     self.exit_event.set()
    #     self.stop_recording()
    #     if self.thread and self.thread.is_alive():
    #         self.thread.join(timeout=2)  # 等待线程安全退出
    #     self.pipeline.stop()
    def stop(self):
        """外部调用的总停止开关"""
        self.exit_event.set() # 停止相机流线程
        self.stop_recording() # 停止并保存视频
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=2)
        # self.pipeline.stop()

if __name__ == "__main__":
    # 创建上下文对象，用于管理所有连接的 RealSense 设备
    ctx = rs.context()

    # 检查是否有设备连接
    if len(ctx.devices) > 0:
        print("Found RealSense devices:")
        for d in ctx.devices:
            # 获取设备的名称和序列号
            name = d.get_info(rs.camera_info.name)
            serial_number = d.get_info(rs.camera_info.serial_number)
            print(f"Device: {name}, Serial Number: {serial_number}")
    else:
        print("No Intel RealSense devices connected")

    # 获取环境变量 PLAYER
    player_value = os.getenv("PLAYER")
    player_value = 1

    # 检查环境变量是否存在且是数字
    if player_value is None:
        raise ValueError("环境变量 PLAYER 未设置")
    try:
        player_value = int(player_value)
    except ValueError:
        raise ValueError("环境变量 PLAYER 必须是一个整数")

    # 根据 PLAYER 的值执行不同的操作
    if player_value == 1:
        print("Player 1")
        cameras = [
            RealSenseCam("244222071389", "left_camera"),
            RealSenseCam("242222071721", "head_camera"),
            RealSenseCam("313522070980", "right_camera"), 
        ]
    # elif player_value == 2:
    #     print("Player 2")
    #     cameras = [
    #         RealSenseCam("250122079815", "left_camera"),
    #         RealSenseCam("048522073543", "head_camera"),
    #         RealSenseCam("030522070109", "right_camera"),
    #     ]
    else:
        raise ValueError("PLAYER 值无效，必须是 1 或 2")

    # 启动所有相机
    for cam in cameras:
        cam.start()

    # 预热相机
    for i in range(20):
        print(f"Warm up: {i}", end="\r")
        for cam in cameras:
            color_image = cam.get_latest_image()
        time.sleep(0.15)

    # 保存每台相机的三张图像
    for i in range(3):
        for cam in cameras:
            color_image = cam.get_latest_image()
            if color_image is not None:
                # 保存图像
                filename = f"{cam.name}_image_{i}.png"
                cv2.imwrite(filename, color_image)
                print(f"Saved image: {filename}")

    # 停止所有相机
    for cam in cameras:
        cam.stop()
