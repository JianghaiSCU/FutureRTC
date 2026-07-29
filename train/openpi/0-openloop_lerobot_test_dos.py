#!/usr/bin/env python3
"""
开环测试：基于 OpenPI API，兼容 LeRobot Dataset 取数据，4-step 抽帧，chunk 版
修复了缺少右侧相机 (cam_wrist_right) 的问题
"""
import argparse
import pathlib
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys

# 导入 LeRobot 数据集
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
sys.path.append(parent_directory)

# 恢复使用 openpi 原生 API
from openpi.models import model as _model
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config


class PI0:
    def __init__(self, train_config_name, checkpoint_path, pi0_step):
        self.train_config_name = train_config_name
        config = _config.get_config(self.train_config_name)

        self.policy = _policy_config.create_trained_policy(
            config,
            checkpoint_path,
        )
        print("loading model success!")
        self.observation_window = None
        self.pi0_step = pi0_step

    def set_language(self, instruction):
        self.instruction = instruction
        print(f"successfully set instruction: {instruction}")

    def update_observation_window(self, img_front, img_left, img_right, state):
        # 【关键修复】加入 img_right，满足双臂策略的三相机需求
        self.observation_window = {
            "state": state,
            "observation/cam_high": img_front,
            "observation/cam_wrist_left": img_left,
            "observation/cam_wrist_right": img_right,
            "prompt": self.instruction,
        }

    def get_action(self):
        assert self.observation_window is not None, "update observation_window first!"
        return self.policy.infer(self.observation_window)["actions"]

    def reset_obsrvationwindows(self):
        self.instruction = None
        self.observation_window = None
        print("successfully unset obs and language intruction")


def reset_model(model):
    model.reset_obsrvationwindows()


# ---------- 健壮地从 LeRobot 读一个 episode 并抽帧 ----------
def load_one_episode_robust(repo_id: str, episode_id: int, step: int = 4, reverse_order: bool = True):
    dataset = LeRobotDataset(repo_id)
    episodes_meta = dataset.meta.episodes
    
    if "dataset_from_index" in episodes_meta:
        from_idx = episodes_meta["dataset_from_index"][episode_id].item()
        to_idx = episodes_meta["dataset_to_index"][episode_id].item()
    elif "length" in episodes_meta:
        lengths = episodes_meta["length"]
        from_idx = int(np.sum(lengths[:episode_id]))
        to_idx = from_idx + int(lengths[episode_id])
    else:
        episode_indices = np.array(dataset.hf_dataset["episode_index"])
        indices = np.where(episode_indices == episode_id)[0]
        if len(indices) == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), ""
        from_idx = int(indices[0])
        to_idx = int(indices[-1] + 1)
        
    T = to_idx - from_idx
    
    idx_obs_forward = list(range(0, T - step, step))
    idx_act_forward = [i + step for i in idx_obs_forward]
    
    if reverse_order:
        idx_obs_relative = idx_obs_forward[::-1]
        idx_act_relative = idx_act_forward[::-1]
    else:
        idx_obs_relative = idx_obs_forward
        idx_act_relative = idx_act_forward
        
    if not idx_obs_relative:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), ""
    
    # 增加 right_imgs 列表
    head_imgs, left_imgs, right_imgs, states, acts = [], [], [], [], []
    task_prompt = ""
    
    for i_obs, i_act in zip(idx_obs_relative, idx_act_relative):
        obs_data = dataset[from_idx + i_obs]
        act_data = dataset[from_idx + i_act]
        
        # 提取 front, left, right 图像
        # front_tensor = obs_data['observation.images.front_image']
        # head_imgs.append((front_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
        head_imgs.append(obs_data['observation.images.front_image'])
        
        # left_tensor = obs_data['observation.images.left_image']
        # left_imgs.append((left_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
        left_imgs.append(obs_data['observation.images.left_image'])


        # right_tensor = obs_data['observation.images.right_image']
        # right_imgs.append((right_tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8))
        right_imgs.append(obs_data['observation.images.right_image'])
        
        states.append(obs_data['observation.state'].numpy())
        acts.append(act_data['observation.state'].numpy()) 
        
        if not task_prompt:
            prompt_val = obs_data['task']
            if hasattr(prompt_val, 'item'):
                task_prompt = str(prompt_val.item())
            elif isinstance(prompt_val, list):
                task_prompt = str(prompt_val[0])
            else:
                task_prompt = str(prompt_val)
            
    return np.array(head_imgs), np.array(left_imgs), np.array(right_imgs), np.array(states), np.array(acts), task_prompt

# ---------- 纵向一张长图 加竖线 ----------
def plot_vertical_stack(trues, preds, out_dir: pathlib.Path, episode_id: int,
                        chunk_size: int = 8):
    D = trues.shape[1]
    T = len(trues)
    plt.figure(figsize=(8, 2.5 * D))
    time = np.arange(T)

    # 所有 chunk 起点
    chunk_starts = np.arange(0, T, chunk_size)

    for d in range(D):
        ax = plt.subplot(D, 1, d + 1)
        # 曲线
        plt.plot(time, trues[:, d], 'b-', label='label')
        plt.plot(time, preds[:, d], 'r--', label='pred')

        # ===== 浅灰色竖线 =====
        for x in chunk_starts:
            ax.axvline(x=x, color='gray', lw=1.0, alpha=0.25)

        plt.ylabel(f'Dim {d}')
        if d == 0:
            plt.legend(loc='upper right')
        if d == D - 1:
            plt.xlabel('step')
        else:
            ax.set_xticklabels([])

    plt.tight_layout()
    save_path = out_dir / f'episode_{episode_id}_all_dims_vertical.png'
    plt.savefig(save_path, dpi=300)
    plt.close()
    print('纵向汇总图已存 →', save_path)




# ---------- 主逻辑 ----------
def openloop_lerobot_chunk(repo_id: str, ckpt: pathlib.Path,
                           episode_id: int = 0, out_dir: pathlib.Path = None,
                           chunk_size: int = 8, step: int = 4, train_config="pi05_aloha_full_base_z1_H"):
    
    # 1. 读数据（解包时加入 right_imgs）
    head_imgs, left_imgs, right_imgs, states, acts, task_prompt = load_one_episode_robust(
        repo_id, episode_id, step, reverse_order=False
    )
    
    if len(acts) == 0:
        print(f"Episode {episode_id} 数据长度不足，跳过。")
        return

    N, D = acts.shape[0], acts.shape[1]
    print(f"episode {episode_id} 共 {N} 帧（已{step}-step抽帧），{D} 维 action，chunk={chunk_size}")
    print(f"train_config: {train_config}")

    # 2. 初始化模型
    model = PI0(train_config, str(ckpt), 50)
    reset_model(model)

    if model.observation_window is None:
        model.set_language(task_prompt)

    preds_list, trues_list = [], []
    t = 0
    
    # 3. 循环推理
    while t < N:
        # 传入三个视角的图像
        model.update_observation_window(head_imgs[t], left_imgs[t], right_imgs[t], states[t])
        
        
        
        # openpi 的 chunk 截取
        chunk_actions = model.get_action()[:model.pi0_step, :14]

        if isinstance(chunk_actions, list):
            chunk_actions = np.array(chunk_actions)
            
        steps_this_chunk = min(chunk_size, N - t)
        chunk_actions = chunk_actions[:steps_this_chunk]
        # print("===========================================")
        # print(states[t])

        for k in range(steps_this_chunk):
            preds_list.append(chunk_actions[k])
            trues_list.append(acts[t + k][:14]) # 真值切片前 14 维对齐
            
            if k < steps_this_chunk - 1:
                # 更新下一步 obs（传入三个视角的图像）
                model.update_observation_window(
                    head_imgs[t + k + 1], 
                    left_imgs[t + k + 1], 
                    right_imgs[t + k + 1],
                    states[t + k + 1]
                )

        t += steps_this_chunk

    preds = np.array(preds_list)
    trues = np.array(trues_list)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_vertical_stack(trues, preds, out_dir, episode_id, chunk_size=chunk_size)
        np.save(out_dir / f"episode_{episode_id}_error_per_dim.npy", preds - trues)

    abs_err = np.abs(preds - trues)
    print("各关节平均绝对误差:", abs_err.mean(axis=0))
    print("整体平均 L2 误差:", np.linalg.norm(preds - trues, axis=1).mean())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="electric_box_button", 
                        help="LeRobot dataset repo_id (e.g., your local directory name or HF hub name)")
    parser.add_argument("--ckpt", type=pathlib.Path, required=True, help="ckpt 路径")
    parser.add_argument("--episode", type=int, default=0, help="episode 号")
    parser.add_argument("--out", type=pathlib.Path, default=None, help="对比图输出目录")
    parser.add_argument("--chunk", type=int, default=16, help="模型一次输出步长")
    parser.add_argument("--step", type=int, default=1, help="抽帧步长（与训练一致）")
    parser.add_argument("--train_config", type=str, default="pi05_electric_box_press_button_finetune_260410", help="openpi config")
    args = parser.parse_args()


    # # 2. 执行测试
    # for i in range(122):
    #     args.episode = i
    #     try:
    #         openloop_lerobot_chunk(
    #             repo_id=args.repo_id, 
    #             ckpt=args.ckpt.expanduser(),
    #             episode_id=args.episode, 
    #             out_dir=args.out, 
    #             chunk_size=args.chunk, 
    #             step=args.step, 
    #             train_config=args.train_config
    #         )
    #     except Exception as e:
    #         print(f"Episode {i} 测试报错: {e}")
    #         break
            
    #     break # 默认测 1 个就退出，如果想测 50 个把这个 break 删掉

    # 如果你想跑 50 个，将循环加上去即可，这里按传参跑单次
    openloop_lerobot_chunk(
        repo_id=args.repo_id, 
        ckpt=args.ckpt.expanduser(),
        episode_id=args.episode, 
        out_dir=args.out, 
        chunk_size=args.chunk, 
        step=args.step, 
        train_config=args.train_config
    )


# uv run 0-openloop_lerobot_test.py --repo_id electric_box_button \
#                                 --ckpt /workspace/ckpt/openpi_w1/checkpoints/pi05_electric_box_press_button_finetune_260410/pi05_electric_box_test/15000  --episode 0  --out ./0-test-show/0-lerobot_cmp_pi05_electric_box_press_button_finetune_260410_15000


# uv run 0-openloop_lerobot_test.py --repo_id electric_box_button \
#                                 --ckpt /workspace/.cache/openpi/openpi-assets/checkpoints/pi05_base  --episode 0  --out ./0-test-show/0-lerobot_cmp_pi05_base

# uv run 0-openloop_lerobot_test.py --repo_id electric_box_button \
#                                 --ckpt /workspace/ckpt/openpi_w1/checkpoints/pi05_electric_box_press_button_finetune_260410/pi05_electric_box_test/20000  --episode 64  --out ./0-test-show/0-lerobot_cmp_pi05_electric_box_press_button_finetune_260410_15000-test

