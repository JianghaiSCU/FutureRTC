from control_joints import *
from read_joints import *
from realsense import *
import os

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
        cameras = [
            RealSenseCam("337322073280", "left_camera"),
            RealSenseCam("337322074191", "head_camera"),
            RealSenseCam("337122072617", "right_camera"),
        ]
        left_can, right_can = "can_left_1", "can_right_1"
    elif player_value == 2:
        print("Player 2")
        cameras = [
            RealSenseCam("250122079815", "left_camera"),
            RealSenseCam("048522073543", "head_camera"),
            RealSenseCam("030522070109", "right_camera"),
        ]
        left_can, right_can = "can_left_2", "can_right_2"
    else:
        raise ValueError("PLAYER 值无效，必须是 1 或 2")

    # ==== Get RGB ====
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
    obs = dict()
    for i in range(3):
        for cam in cameras:
            color_image = cam.get_latest_image()
            if color_image is not None:
                # 保存图像
                obs[cam] = color_image
                # filename = f"{cam.name}_image_{i}.png"
                # cv2.imwrite(filename, color_image)
                # print(f"Saved image: {filename}")

    # ==== Get Joint ====
    reader = JointReader(left_can=left_can, right_can=right_can)
    print(reader.get_joint_value())

    # ==== Deploy Action ====
    controller = ControlJoints(left_can=left_can, right_can=right_can)
    for i in range(10):
        positions = [0] * 14
        controller.control(positions)
        time.sleep(0.1)
