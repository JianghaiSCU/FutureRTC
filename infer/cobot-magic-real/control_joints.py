#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 注意: 示例代码无法直接运行，需要安装SDK后才能运行
from typing import Optional
import time, os, sys
import socket

from piper_sdk.piper_sdk import *  # Assuming this is a placeholder for actual SDK imports
import threading

class ControlJoints:
    def __init__(self, left_can="can_left_1", right_can="can_right_1"):
        self.left_arm = ControlSingleArm(left_can)
        self.right_arm = ControlSingleArm(right_can)

    def control(self, joints):
        self.left_arm.control(joints[:7])
        self.right_arm.control(joints[7:])
        time.sleep(0.03)

class ControlSingleArm:
    def __init__(self, can_name="can_left_1"):
        host = "127.0.0.1"
        if can_name == "can_left_1":
            port = 9990
        elif can_name == "can_right_1":
            port = 9991
        elif can_name == "can_left_2":
            port = 9992
        elif can_name == "can_right_2":
            port = 9993
        self.can_name = can_name
        self.host = host
        self.port = port
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connect_to_server()

    def connect_to_server(self):
        try:
            self.client_socket.connect((self.host, self.port))
            print(f"Connected to server at {self.host}:{self.port}")
        except Exception as e:
            print(f"Failed to connect to server: {e}")
            sys.exit(1)

    def control(self, joints):  # joints: 7 dim
        try:
            # Convert the joints list to a comma-separated string
            joints_str = ",".join(map(str, joints))
            # print(f"Sending joints: {joints_str}")
            # Send the joints data to the server
            self.client_socket.sendall(joints_str.encode())
        except Exception as e:
            print(f"Error sending data: {e}")

    def close_connection(self):
        self.client_socket.close()
        print("Connection closed.")


if __name__ == "__main__":
    # 获取环境变量 PLAYER
    player_value = os.getenv("PLAYER")

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
        left_can, right_can = "can_left_1", "can_right_1"
    elif player_value == 2:
        print("Player 2")
        left_can, right_can = "can_left_2", "can_right_2"
    else:
        raise ValueError("PLAYER 值无效，必须是 1 或 2")


    left_can_name = left_can
    right_can_name = right_can

    controller = ControlJoints(left_can_name, right_can_name)

    position = [0, 0, 0, 0, 0, 0, 0]
    count = 0

    while True:
        print(f"Count: {count}")
        if count == 0:
            print("1-----------")
            position = [0, 0, 0, 0, 0, 0, 0.01]
        elif count == 100:
            print("2-----------")
            # position = [0.2, 0.2, -0.2, 0.3, -0.2, 0.5, 0.08]
            position = [0.2, 0.2, -0.2, 0.3, -0.2, 0.5, 0.09]
        elif count == 200:
            print("1-----------")
            position = [0, 0, 0, 0, 0, 0, 0.01]
            count = 0
        elif count==201:
            break

        joints_list = position + position  # Duplicate for left and right arms
        controller.control(joints_list)
        count += 1
