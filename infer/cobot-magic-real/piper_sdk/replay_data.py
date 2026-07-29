from control_joints import *
from read_joints import *
from realsense import *
import os, json, time

if __name__ == "__main__":

    def list_and_sort_json_files(directory):
        # 获取目录中的所有文件
        files = os.listdir(directory)
        # 筛选出所有 .json 文件
        json_files = [file for file in files if file.endswith(".json")]
        # 对文件名进行排序
        json_files.sort()
        return json_files

    # 示例用法
    directory = "/data/gjx/Desktop/episode1 (1)"  # 替换为你的目录路径
    left_action_path = os.path.join(directory, "arm/jointState/puppetLeft")
    right_action_path = os.path.join(directory, "arm/jointState/puppetRight")

    left_jsons = list_and_sort_json_files(left_action_path)
    right_jsons = list_and_sort_json_files(right_action_path)

    # ==== Deploy Action ====
    controller = ControlJoints()
    ctrl_list = []
    for i in range(0, min(len(left_jsons), len(right_jsons)), 10):
        print(i)
        with open(os.path.join(left_action_path, left_jsons[i]), "r") as file:
            left_data = json.load(file)
        with open(os.path.join(right_action_path, right_jsons[i]), "r") as file:
            right_data = json.load(file)

        positions = left_data["position"] + right_data["position"]
        ctrl_list.append(positions)

    for i in range(len(ctrl_list)):
        controller.control(ctrl_list[i])
        time.sleep(0.005)
