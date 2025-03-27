#!/bin/bash
# 硬件检测演示用例

# 加载模块
source ../modules/hardware/hardware_check.sh

# 执行硬件检测
collect_hardware_info
collect_gpu_info

echo "检测完成，结果保存在hardware_demo.log"