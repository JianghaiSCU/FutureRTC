#!/usr/bin/env python3
# -*-coding:utf8-*-
# 注意demo无法直接运行，需要pip安装sdk后才能运行
from typing import (
    Optional,
)
import time, os
from piper_sdk.piper_sdk import *


class ControlJoints:
    def __init__(self, left_can="can_left_1", right_can="can_right_1"):
        left_piper = C_PiperInterface_V2(left_can)
        left_piper.ConnectPort()
        right_piper = C_PiperInterface_V2(right_can)
        right_piper.ConnectPort()
        while (not left_piper.EnablePiper()) or (not right_piper.EnablePiper()):
            time.sleep(0.01)
        self.left_piper = left_piper
        self.right_piper = right_piper

    def control(self, joints):
        factor = 57295.7795  # 1000*180/3.1415926

        def update(list_val, factor):
            return [float(x) * factor for x in list_val]

        joints_list = list(joints)
        left_joints_list = update(joints_list[:6], factor)
        left_gripper_val = int(float(joints_list[6]) * 1000 * 1000)
        right_joints_list = update(joints_list[7:13], factor)
        right_gripper_val = int(float(joints_list[13]) * 1000 * 1000)

        left_joints_list = [int(x) for x in left_joints_list]
        right_joints_list = [int(x) for x in right_joints_list]

        self.left_piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        self.left_piper.JointCtrl(
            left_joints_list[0],
            left_joints_list[1],
            left_joints_list[2],
            left_joints_list[3],
            left_joints_list[4],
            left_joints_list[5],
        )
        self.left_piper.GripperCtrl(abs(left_gripper_val), 1000, 0x01, 0)
        self.right_piper.MotionCtrl_2(0x01, 0x01, 100, 0x00)
        self.right_piper.JointCtrl(
            right_joints_list[0],
            right_joints_list[1],
            right_joints_list[2],
            right_joints_list[3],
            right_joints_list[4],
            right_joints_list[5],
        )
        self.right_piper.GripperCtrl(abs(right_gripper_val), 1000, 0x01, 0)


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

    position = [0, 0, 0, 0, 0, 0, 0]
    count = 0

    controller = ControlJoints(left_can=left_can, right_can=right_can)

    while True:
        print(count)
        count = count + 1
        if count == 0:
            print("1-----------")
            position = [0, 0, 0, 0, 0, 0, 0]
            # position = [0.2,0.2,-0.2,0.3,-0.2,0.5,0.08]
        elif count == 10:
            print("2-----------")
            position = [0.2, 0.2, -0.2, 0.3, -0.2, 0.5, 0.08]
            # position = [0,0,0,0,0,0,0]
            # position = [-8524,104705,-78485,-451,-5486,29843,0]
        elif count == 20:
            print("1-----------")
            position = [0, 0, 0, 0, 0, 0, 0]
            # position = [0.2,0.2,-0.2,0.3,-0.2,0.5,0.08]
            count = 0

        joint_0 = position[0]
        joint_1 = position[1]
        joint_2 = position[2]
        joint_3 = position[3]
        joint_4 = position[4]
        joint_5 = position[5]
        joint_6 = position[6]
        joints_list = [
            joint_0,
            joint_1,
            joint_2,
            joint_3,
            joint_4,
            joint_5,
            abs(joint_6),
            joint_0,
            joint_1,
            joint_2,
            joint_3,
            joint_4,
            joint_5,
            abs(joint_6),
        ]
        controller.control(joints_list)

        time.sleep(0.005)
        pass
