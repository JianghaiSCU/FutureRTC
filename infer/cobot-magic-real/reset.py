from control_joints import *
from control_joints import *
import time, os, sys

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
    # ==== Deploy Action ====
    controller = ControlJoints(left_can=left_can, right_can=right_can)

    positions = [0] * 14
    positions[6], positions[13] = 0.1, 0.1
    controller.control(positions)
    time.sleep(0.1)

    positions = [0] * 14
    positions[6], positions[13] = 0, 0
    controller.control(positions)
    time.sleep(0.1)

    # positions = [0.3537992,   0.08769099, -0.00441333, -0.0866269,   0.2551883,   0.19741374,  0.,         
    #              -0.01215847,  0.00462266, -0.00350624,  0.01404242,  0.2187652, -0.12013683,  0.,       ]
    # controller.control(positions)
    # time.sleep(1)



