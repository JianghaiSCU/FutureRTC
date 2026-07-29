import pyrealsense2 as rs
import numpy as np
import threading
from collections import deque
import cv2
import time
import os


class RealSenseCam:
    def __init__(self, serial_number, name):
        self.serial_number = serial_number
        self.name = name
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(serial_number)
        # 只启用彩色图像流
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # 使用双端队列替换队列，简化帧管理
        self.frame_buffer = deque(maxlen=1)  # 仅保留最新一帧
        self.keep_running = False
        self.thread = None
        self.exit_event = threading.Event()

    def start(self):
        self.keep_running = True
        self.exit_event.clear()
        self.pipeline.start(self.config)
        self.thread = threading.Thread(target=self._update_frames)
        self.thread.daemon = True  # 设置为守护线程
        self.thread.start()

    def _update_frames(self):
        try:
            while not self.exit_event.is_set():
                # 等待彩色帧数据（超时5秒）
                frames = self.pipeline.wait_for_frames(5000)
                color_frame = frames.get_color_frame()

                if color_frame:
                    # 转换为NumPy数组并存储
                    color_image = np.asanyarray(color_frame.get_data())[:, :, ::-1]
                    self.frame_buffer.append(color_image)  # 保留最新帧
        except Exception as e:
            print(f"Error from {self.name} camera: {e}")
        finally:
            self.pipeline.stop()

    def get_latest_image(self):
        if self.frame_buffer:
            return self.frame_buffer[-1]  # 返回最新一帧
        return None

    def stop(self):
        self.exit_event.set()
        self.keep_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2)  # 等待线程安全退出
        self.pipeline.stop()


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
