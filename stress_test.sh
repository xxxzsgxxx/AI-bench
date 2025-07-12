#!/bin/bash
# 参数校验
DURATION=${1:-0}
if ! [[ "$DURATION" =~ ^[0-9]+$ ]]; then
    echo "错误：持续时间参数必须是整数（单位：分钟）"
    exit 1
fi

apt install stress-ng fio iperf3 dstat

# 日志目录设置
LOG_DIR="stress_logs/$(date +%Y%m%d_%H%M)"
mkdir -p "$LOG_DIR"
declare -a PIDS

# 硬件信息快照
{
    lscpu
    lspci -vvv
    nvidia-smi -q
    smartctl -a /dev/nvme0n1
    dmidecode -t memory
} > "$LOG_DIR/hardware_info.txt"

# 压力测试函数
start_stress() {
    # GPU压力（CUDA + OpenGL）
    stress-ng --cuda $(nvidia-smi -L | wc -l) --opengl $(nproc) -t ${DURATION}m &
    PIDS+=($!)

    # CPU+内存压力
    stress-ng --cpu $(nproc) --vm $(nproc) --vm-bytes 95% -t ${DURATION}m &
    PIDS+=($!)

    # 磁盘压力（随机读写）
    fio --name=disk_test --ioengine=libaio --rw=randrw --bs=4k --direct=1 \
        --numjobs=4 --size=16G --runtime=${DURATION}m --time_based --group_reporting &
    PIDS+=($!)

    # 网络压力（需预先启动iperf3服务端）
    iperf3 -c <SERVER_IP> -t ${DURATION} -b 0 -P 8 &
    PIDS+=($!)
}

# 监控采集函数
start_monitor() {
    # GPU监控（每秒采样）
    while true; do
        nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw \
            --format=csv >> "$LOG_DIR/gpu.csv"
        sleep 1
    done &
    PIDS+=($!)

    # 系统监控（dstat每秒采样）
    dstat -tcmnd --disk-util --output "$LOG_DIR/system.csv" 1 0 &
    PIDS+=($!)

    # IPMI传感器监控（每5秒采样）
    while true; do
        ipmitool sensor >> "$LOG_DIR/ipmi.log"
        sleep 5
    done &
    PIDS+=($!)
}

# 信号捕获
cleanup() {
    echo "正在终止所有进程..."
    kill -9 "${PIDS[@]}" 2>/dev/null
    pkill -f "stress-ng|fio|iperf3"
    exit 0
}
trap cleanup EXIT INT TERM

# 执行测试
start_stress
start_monitor

# 持续时间控制
if (( DURATION > 0 )); then
    echo "压力测试将持续 $DURATION 分钟..."
    sleep ${DURATION}m
    cleanup  # 新增自动清理调用
else
    echo "压力测试持续运行，按 Ctrl+C 终止..."
    wait
fi