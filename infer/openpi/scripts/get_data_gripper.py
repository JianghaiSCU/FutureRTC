import pandas as pd
import numpy as np

# 读取你的 parquet 数据文件
file_path = "/root/.cache/spray_painting/data/chunk-000/episode_000000.parquet"
df = pd.read_parquet(file_path)

# 因为 parquet 里的数组是以 object 形式存储的，我们需要把它们堆叠成矩阵
eff_states = np.vstack(df['observation.state.effector.position'].values)
eff_actions = np.vstack(df['action.effector.position'].values)

# 打印最小值和最大值 (axis=0 表示沿着时间维度求每一列的极值)
print(f"抓夹状态 (State) - 最小值: {eff_states.min(axis=0)}, 最大值: {eff_states.max(axis=0)}")
print(f"抓夹动作 (Action) - 最小值: {eff_actions.min(axis=0)}, 最大值: {eff_actions.max(axis=0)}")