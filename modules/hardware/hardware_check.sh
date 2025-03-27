#!/bin/bash
# 硬件信息采集模块

collect_hardware_info() {
    echo "[硬件模块] 采集基础硬件信息..."
    lshw -short 2>&1
    lscpu
    dmidecode -t bios 2>&1
}

collect_gpu_info() {
    echo "[GPU模块] 检测显卡设备..."
    nvidia-smi -q 2>&1 || echo "未检测到NVIDIA显卡"
    rocm-smi 2>&1 || echo "未检测到AMD显卡"
}