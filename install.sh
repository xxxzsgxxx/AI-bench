#!/bin/bash
# 环境初始化模块
set -e

# 加载可视化组件
source "$(dirname "$0")/cli/visualization.sh"

# 基础工具链安装
apt_update() {
    print_step 1 "更新软件源缓存"
    sudo apt-get update -qq
    echo "✅ 软件源更新完成"
}

install_essentials() {
    print_step 2 "安装基础开发工具"
    sudo apt-get install -y -qq \
        build-essential \
        cmake \
        git \
        python3-pip \
        nvidia-cuda-toolkit
    echo "✅ 基础工具安装完成"
}

install_benchmark_tools() {
    print_step 3 "安装基准测试工具"
    sudo apt-get install -y -qq \
        fio \
        iperf3 \
        stress-ng \
        nvtop
    echo "✅ 测试工具安装完成"
}

setup_python_env() {
    print_step 4 "配置Python虚拟环境"
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -q \
        tensorflow \
        torch \
        psutil \
        gpustat
    echo "✅ Python环境配置完成"
}

main() {
    # 操作系统兼容性检查
    print_step 0 "验证操作系统类型"
    if ! grep -qEi '^(ID=ubuntu|ID=debian)$' /etc/os-release; then
        show_error "本工具仅支持在Ubuntu/Debian系统运行"
    fi

    show_progress 4
    apt_update
    install_essentials
    install_benchmark_tools
    setup_python_env
    echo "\n🎉 环境初始化完成"
}

main "$@"