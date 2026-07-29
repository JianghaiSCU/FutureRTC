import numpy as np
import time, os
from multiprocessing import shared_memory, Lock, Process
from piper_sdk.piper_sdk import *

def joint_reader_worker(can_name, shm_name, lock, is_left: bool):
    piper = C_PiperInterface_V2(
        can_name=can_name,
        judge_flag=False,
        can_auto_init=True,
        dh_is_offset=1,
        start_sdk_joint_limit=False,
        start_sdk_gripper_limit=False,
    )
    piper.ConnectPort()

    shm = shared_memory.SharedMemory(name=shm_name)
    data = np.ndarray((7,), dtype='float32', buffer=shm.buf)

    factor = 57295.7795

    def extract(joint_msg, gripper_msg):
        joints = joint_msg.joint_state
        joint_list = [
            joints.joint_1,
            joints.joint_2,
            joints.joint_3,
            joints.joint_4,
            joints.joint_5,
            joints.joint_6,
        ]
        gripper = gripper_msg.gripper_state.grippers_angle

        for i in range(6):
            joint_list[i] = joint_list[i] / factor
        gripper = gripper / (1000 * 1000)

        return joint_list + [gripper]
    count = 0
    while True:
        count = count+1
        if(count > 500):
            # print(piper.GetCanFps())
            count = 0
        try:
            joint_msg = piper.GetArmJointMsgs()
            gripper_msg = piper.GetArmGripperMsgs()
            values = extract(joint_msg, gripper_msg)

            with lock:
                data[:] = values
        except Exception as e:
            print(f"[{can_name}] error: {e}")
        time.sleep(0.002)


class JointReader:
    def __init__(self, left_can, right_can):
        self.left_shm = shared_memory.SharedMemory(create=True, size=7 * 4, name=f"left_data_{left_can}")
        self.right_shm = shared_memory.SharedMemory(create=True, size=7 * 4, name=f"right_data_{right_can}")

        self.left_lock = Lock()
        self.right_lock = Lock()

        self.left_array = np.ndarray((7,), dtype='float32', buffer=self.left_shm.buf)
        self.right_array = np.ndarray((7,), dtype='float32', buffer=self.right_shm.buf)

        self.left_proc = Process(
            target=joint_reader_worker,
            args=(left_can, f"left_data_{left_can}", self.left_lock, True),
        )
        self.right_proc = Process(
            target=joint_reader_worker,
            args=(right_can, f"right_data_{right_can}", self.right_lock, False),
        )

        self.left_proc.start()
        self.right_proc.start()

    def get_joint_value(self):
        with self.left_lock:
            left = self.left_array.copy()
        with self.right_lock:
            right = self.right_array.copy()
        return list(left) + list(right)

    def close(self):
        self.left_proc.terminate()
        self.right_proc.terminate()
        self.left_proc.join()
        self.right_proc.join()
        self.left_shm.close()
        self.left_shm.unlink()
        self.right_shm.close()
        self.right_shm.unlink()

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

    reader = JointReader(left_can=left_can, right_can=right_can)
    
    while True:
        print(reader.get_joint_value())
        time.sleep(0.01)
