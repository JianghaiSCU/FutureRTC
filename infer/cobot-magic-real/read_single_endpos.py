import numpy as np
import time, os
from multiprocessing import shared_memory, Lock, Process
from piper_sdk.piper_sdk import *

def endpos_reader_worker(can_name, shm_name, lock, is_left: bool):
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

    factor = 1000 * 1000
    # X,Y,Z单位0.001mm
    # RX,RY,RZ单位0.001度
    def extract(endpos_msg, gripper_msg):
        endpos = endpos_msg.end_pose
        endpos_list = [
            endpos.X_axis,
            endpos.Y_axis,
            endpos.Z_axis,
            endpos.RX_axis,
            endpos.RY_axis,
            endpos.RZ_axis,
        ]
        gripper = gripper_msg.gripper_state.grippers_angle

        for i in range(6):
            endpos_list[i] = endpos_list[i] / factor
        gripper = gripper / (1000 * 1000)

        return endpos_list + [gripper]
    count = 0
    while True:
        count = count+1
        if(count > 500):
            # print(piper.GetCanFps())
            count = 0
        try:
            endpos_msg = piper.GetArmEndPoseMsgs()
            gripper_msg = piper.GetArmGripperMsgs()
            values = extract(endpos_msg, gripper_msg)

            with lock:
                data[:] = values
        except Exception as e:
            print(f"[{can_name}] error: {e}")
        time.sleep(0.002)


class EndposReader:
    def __init__(self, left_can):
        self.left_shm = shared_memory.SharedMemory(create=True, size=7 * 4, name=f"left_data_{left_can}")

        self.left_lock = Lock()

        self.left_array = np.ndarray((7,), dtype='float32', buffer=self.left_shm.buf)

        self.left_proc = Process(
            target=endpos_reader_worker,
            args=(left_can, f"left_data_{left_can}", self.left_lock, True),
        )

        self.left_proc.start()

    def get_endpos_value(self):
        with self.left_lock:
            left = self.left_array.copy()
        # with self.right_lock:
        #     right = self.right_array.copy()
        return list(left)

    def close(self):
        self.left_proc.terminate()
        self.left_proc.join()
        self.left_shm.close()
        self.left_shm.unlink()


if __name__ == "__main__":
    
    left_can = "can_right"

    reader = EndposReader(left_can=left_can)
    
    while True:
        print(reader.get_endpos_value())
        time.sleep(0.01)
