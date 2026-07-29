# Install 

```
pip install pyrealsense2
cd piper_sdk
pip install -e .
```

# 设置Can
```
# 修改can_config.sh 109行附近的can信息
# 以及95行的can模块数量，一套机械的话为2，两套臂控制为4
# 可以参考 https://github.com/agilexrobotics/piper_sdk/tree/0_3_0_beta?tab=readme-ov-file#21-find-can-modules，利用piper_sdk/piper_sdk中的final_all_can_port.sh脚本找到左右机械臂的port，然后修改一下USB_PORTS中的key，如果你只有一套臂进行控制，请只保留can_left_1和can_right_1

bash can_config.sh
```

# SET PLAYER
因为真机比赛的时候使用一台主机控制多台机器，所以需要利用系统PLAYER变量来指定控制的机械臂以及相机，每次开启终端的时候都需要重新设置，请不要修改`~/.bashrc`文件。如果你的本地只有一套机械臂设备，请设置${id}为1即可，同时`can_config.sh`中应该也只有1。
```
# ${id}可以是1或者2
source set_player ${id}
```

# Run
在一个终端开启控制的监听server
```
source set_player ${id}
bash run_server.sh
```

接下来开启部署代码运行，会给server发控制信号
`demo_deploy.py`作为部署参考，请保证你的策略可以读取`instruction.txt`，以支持评测时赛方进行修改
```
# reset arm position
python reset.py
# deploy，请完善脚本，使得直接运行以下指令可以开启部署，其中第一行默认调用reset.py使机械臂回0位
bash deploy.sh
```

# 在你的设备上部署
需要留意can口需要改，这个在前面部分有介绍，同时对于RealSense的序列号也需要修改，可以全局找找`RealSenseCam`的调用，将其中对应`player`的序列号进行修改

