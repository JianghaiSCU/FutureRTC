import numpy as np
import time, os
from multiprocessing import shared_memory, Lock, Process
from piper_sdk.piper_sdk import *



from scipy.spatial.transform import Rotation as R

def euler_to_6d(roll: float,
                pitch: float,
                yaw: float,
                seq: str = 'ZYX',
                degrees: bool = True) -> np.ndarray:
    """
    欧拉角 → 6D 旋转表示（旋转矩阵的前两列）
    参数
    ----
    roll, pitch, yaw : float
        欧拉角，默认单位“度”
    seq : str
        旋转顺序，'ZYX'、'XYZ'、'ZYZ' … 任意 scipy 支持的三位字符串
    degrees : bool
        True  表示输入是度
        False 表示输入是弧度
    返回
    ----
    np.ndarray, shape=(6,), dtype=float32
        顺序 = [r11, r21, r31, r12, r22, r32]  （即 R[:,0] 和 R[:,1]）
    """
    # 1. 欧拉角 → scipy Rotation 对象（一行现成的函数）
    rot = R.from_euler(seq=seq, angles=[roll, pitch, yaw], degrees=degrees)
    # 2. Rotation 对象 → 3×3 旋转矩阵（现成的函数）
    R_mat = rot.as_matrix()          # shape=(3,3)
    # 3. 取前两列，拉平 → 6D
    sixd = R_mat[:, :2].T.ravel()    # 先转置再 ravel，得到顺序与论文一致
    return sixd.astype(np.float32)


def sixd_to_euler(sixd: np.ndarray,
                  seq: str = 'ZYX',
                  degrees: bool = True) -> np.ndarray:
    """
    6D 旋转表示 → 欧拉角
    参数
    ----
    sixd : array-like, shape=(6,)
        6D 向量，顺序为 [r11,r21,r31,r12,r22,r32] （即 R[:,0] 和 R[:,1]）
    seq : str
        目标欧拉角顺序，'ZYX'/'XYZ'/... 任意 scipy 支持的三字符
    degrees : bool
        True  返回角度制；False 返回弧度制
    返回
    ----
    np.ndarray, shape=(3,)
        数组顺序与 seq 对应，例如 seq='ZYX' 时返回 [roll, pitch, yaw]
    """
    sixd = np.asarray(sixd, dtype=float)
    if sixd.shape != (6,):
        raise ValueError("6D input must be shape (6,)")

    # 1. 取出前两列
    a1, a2 = sixd[:3], sixd[3:]          # shape=(3,)
    # 2. 施密特正交化，得到标准正交基
    b1 = a1 / np.linalg.norm(a1)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 /= np.linalg.norm(b2)
    # 3. 第三列 = 前两列的叉积，保证右手系
    b3 = np.cross(b1, b2)
    # 4. 组装成 3×3 旋转矩阵
    R_mat = np.stack([b1, b2, b3], axis=1)  # shape=(3,3)
    # 5. 旋转矩阵 → 欧拉角（现成的函数）
    euler = R.from_matrix(R_mat).as_euler(seq, degrees=degrees)
    return euler

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

    factor = 1000

    def extract(joint_ctrl, gripper_ctrl,FK_ctrl):
        joints = joint_ctrl.joint_ctrl
        joint_list = [
            joints.joint_1,
            joints.joint_2,
            joints.joint_3,
            joints.joint_4,
            joints.joint_5,
            joints.joint_6,
        ]
        gripper = gripper_ctrl.gripper_ctrl.grippers_angle
        fk_pos = FK_ctrl[-1]
        # fk_pos = __piper_fk.CalFK(joint_list)
        # print("___________________")
        

        

        for i in range(6):
            joint_list[i] = joint_list[i] / factor

        for i in range(3):
            fk_pos[i] = fk_pos[i] / factor
        gripper = gripper / (1000 * 1000)
        # if not fk_pos == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]:
        # seq = "XYZ"
        # sixd = euler_to_6d(fk_pos[3], fk_pos[4], fk_pos[5], seq=seq, degrees=True)
        # tran_euler = sixd_to_euler(sixd, seq=seq, degrees=True)


        print(fk_pos)


        return fk_pos + [gripper]
    count = 0
    while True:
        count = count+1
        if(count > 500):
            # print(piper.GetCanFps())
            count = 0
        try:
            joint_ctrl = piper.GetArmJointCtrl()
            gripper_ctrl = piper.GetArmGripperCtrl()
            FK_ctrl = piper.GetFK(mode="control")
            # __piper_fk = piper.__piper_fk
            values = extract(joint_ctrl, gripper_ctrl,FK_ctrl)

            with lock:
                data[:] = values
        except Exception as e:
            print(f"[{can_name}] error: {e}")
        time.sleep(0.002)


class JointReader:
    def __init__(self, left_can):
        self.left_shm = shared_memory.SharedMemory(create=True, size=7 * 4, name=f"left_data_{left_can}")

        self.left_lock = Lock()

        self.left_array = np.ndarray((7,), dtype='float32', buffer=self.left_shm.buf)


        self.left_proc = Process(
            target=joint_reader_worker,
            args=(left_can, f"left_data_{left_can}", self.left_lock, True),
        )


        self.left_proc.start()


    def get_joint_value(self):
        with self.left_lock:
            left = self.left_array.copy()

        return list(left)

    def close(self):
        self.left_proc.terminate()

        self.left_proc.join()

        self.left_shm.close()
        self.left_shm.unlink()


if __name__ == "__main__":

    
    left_can = "can_left"

    reader = JointReader(left_can=left_can)
    
    while True:
        end_pos_with_grip = reader.get_joint_value()
        time.sleep(0.1)


# # __UpdatePiperCtrlFK
# [[0, -0.0, 123.0, 0.0, 0.0, -3.543], 
#  [0.0, 0.0, 123.0, 90.0, -6.956999999999976, 176.45700000000002], 
#  [-282.39065275622704, 17.48448664867087, 157.5240910458835, -89.99999999999996, 84.91199999999999, -3.542999999999961], 
#  [-35.05164805923067, 2.1702562266980596, 201.65541781136753, -33.080782163442315, 83.92428718875657, -36.77155456108628], 
#  [-35.05164805923067, 2.1702562266980596, 201.65541781136753, 102.64501714777687, 74.69819386322818, -171.62778375442434], 
#  [52.581188919757864, -5.068057618279499, 178.22269078919558, 149.06162286343087, 72.52935256297943, 143.13357045549333]]


# # GetFK
# [[0, -0.0, 123.0, 0.0, 0.0, -3.543], 
#  [0.0, 0.0, 123.0, 90.0, -6.956999999999976, 176.45700000000002], 
#  [-282.39065275622704, 17.48448664867087, 157.5240910458835, -89.99999999999996, 84.91199999999999, -3.542999999999961], 
#  [-35.05164805923067, 2.1702562266980596, 201.65541781136753, -33.080782163442315, 83.92428718875657, -36.77155456108628], 
#  [-35.05164805923067, 2.1702562266980596, 201.65541781136753, 102.64501714777687, 74.69819386322818, -171.62778375442434], 
#  [0.0, -0.0, 0.0, 0.0, 0.0, 0.0]]

# # init end_pos
# [55.68870655656289, -3.416100385076331, 204.24551476375927, 57.96880751327123, 86.9518551438538, 54.514041447296584]
