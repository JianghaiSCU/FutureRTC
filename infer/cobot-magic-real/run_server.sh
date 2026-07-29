#!/bin/bash

# 获取环境变量 PLAYER
player_value=$(printenv PLAYER)

# 检查环境变量是否存在且是数字
if [ -z "$player_value" ]; then
    echo "环境变量 PLAYER 未设置" >&2
    exit 1
fi

if ! [[ "$player_value" =~ ^[0-9]+$ ]]; then
    echo "环境变量 PLAYER 必须是一个整数" >&2
    exit 1
fi

# 根据 PLAYER 的值执行不同的操作
if [ "$player_value" -eq 1 ]; then
    echo "Player 1"
    left_can="can_left_1"
    right_can="can_right_1"
    left_port="9990"
    right_port="9991"
elif [ "$player_value" -eq 2 ]; then
    echo "Player 2"
    left_can="can_left_2"
    right_can="can_right_2"
    left_port="9992"
    right_port="9993"
else
    echo "PLAYER 值无效，必须是 1 或 2" >&2
    exit 1
fi

# 输出结果（可以根据需要进行其他操作）
echo "Left CAN: $left_can"
echo "Right CAN: $right_can"
echo "Left Port: $left_port"
echo "Right Port: $right_port"

# 函数：终止占用指定端口的进程
kill_port() {
    local port=$1
    if lsof -i :$port > /dev/null; then
        echo "端口 $port 已被占用，正在终止相关进程..."
        lsof -i :$port | awk 'NR!=1 {print $2}' | xargs kill -9
        echo "端口 $port 已释放"
    else
        echo "端口 $port 未被占用"
    fi
}

# 释放目标端口
kill_port $left_port
kill_port $right_port

# 启动两个 Python 脚本进程，分别控制左臂和右臂
# 假设 Python 脚本的名称为 control_arm_server.py

# 启动左臂控制进程
python3 control_arm_server.py "$left_can" "127.0.0.1" "$left_port" & LEFT_PID=$!

# 启动右臂控制进程
python3 control_arm_server.py "$right_can" "127.0.0.1" "$right_port" & RIGHT_PID=$!

wait $LEFT_PID $RIGHT_PID