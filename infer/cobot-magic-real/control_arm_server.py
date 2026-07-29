#!/usr/bin/env python3
# -*-coding:utf8-*-
# 注意demo无法直接运行，需要pip安装sdk后才能运行
from typing import (
    Optional,
)
import time, os
import socket
from piper_sdk.piper_sdk import *
import sys


class ControlArmServer:
    def __init__(self, can_name="can_left_1", host="127.0.0.1", port=12345):
        self.piper = C_PiperInterface_V2(can_name)
        self.piper.ConnectPort()
        while not self.piper.EnablePiper():
            time.sleep(0.01)

        # 初始化socket
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        print(f"Server listening on {self.host}:{self.port}")

    def control(self, joints):
        factor = 57295.7795  # 1000*180/3.1415926

        def update(list_val, factor):
            return [float(x) * factor for x in list_val]

        joints_list = list(joints)
        arm_joints_list = update(joints_list[:6], factor)
        arm_joints_list = [int(x) for x in arm_joints_list]
        gripper_val = int(float(joints_list[6]) * 1000 * 1000)
        self.piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        self.piper.JointCtrl(
            arm_joints_list[0],
            arm_joints_list[1],
            arm_joints_list[2],
            arm_joints_list[3],
            arm_joints_list[4],
            arm_joints_list[5],
        )
        self.piper.GripperCtrl(abs(gripper_val), 1000, 0x01, 0)
        # time.sleep(0.05)

    def listen_and_control(self):
        while True:
            print("Waiting for connection...")
            client_socket, addr = self.server_socket.accept()
            print(f"Connection from {addr}")
            try:
                while True:
                    data = client_socket.recv(1024)
                    if not data:
                        break
                    joints = data.decode().strip().split(",")
                    print(f"Received joints: {joints}")
                    self.control(joints)
            except Exception as e:
                print(f"Error: {e}")
            finally:
                client_socket.close()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 control_arm.py <CAN_NAME> <HOST> <PORT>")
        sys.exit(1)

    can_name = sys.argv[1]
    host = sys.argv[2]
    port = int(sys.argv[3])

    arm = ControlArmServer(can_name, host, port)
    arm.listen_and_control()
