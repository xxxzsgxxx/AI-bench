#!/bin/bash
# 环境初始化模块
set -e

# 基础工具链安装
apt_update() {
    echo "[1/4] 更新软件源缓存..."
    sudo apt-get update -qq
}

install_essentials() {
    echo "[2/4] 安装基础开发工具..."
    sudo apt-get install -y -qq \
        build-essential \
        cmake \
        git \
        python3-pip \
        nvidia-cuda-toolkit  # 包含CUDA开发环境
}

install_benchmark_tools() {
    echo "[3/4] 安装基准测试工具..."
    sudo apt-get install -y -qq \
        fio \
        iperf3 \
        stress-ng \
        nvtop  # GPU监控工具
}

setup_python_env() {
    echo "[4/4] 配置Python虚拟环境..."
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -q \
        tensorflow \
        torch \
        psutil \
        gpustat
}

main() {
    apt_update
    install_essentials
    install_benchmark_tools
    setup_python_env
    echo "环境初始化完成"
}

main "$@"