#!/bin/bash
################################################################################
# 整机全量压力测试脚本 — 单元测试模式
# 覆盖：GPU / CPU / 内存 / 硬盘 / 网卡
# 采集系统状态数据到 CSV，过程数据留存到 LOG
################################################################################
set -eo pipefail

#============================== 配置参数 ==============================
DURATION=${DURATION:-10}                # 每部件测试时长（分钟）
MONITOR_INTERVAL=${MONITOR_INTERVAL:-2} # 监控采样间隔（秒）
LOG_DIR="stress_$(date +%Y%m%d_%H%M%S)"
ALL_COMPONENTS=("gpu" "cpu" "mem" "disk" "net")

#============================== 颜色输出 ==============================
RED='\033[0;31m'; GREEN='\033[0;32m'
YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
pass()  { echo -e "${GREEN}[PASS]${NC} $1"; }
fail()  { echo -e "${RED}[FAIL]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
header(){ echo -e "\n${BLUE}==== $1 ====${NC}"; }

#============================== 全局变量 ==============================
MONITOR_PID=""
COMPONENTS=()

#============================== 工具检测 ==============================
check_prereqs() {
    local missing=()
    for cmd in stress-ng nvidia-smi fio iperf3; do
        command -v "$cmd" &>/dev/null || missing+=("$cmd")
    done
    if [ ${#missing[@]} -gt 0 ]; then
        warn "缺少工具: ${missing[*]}"
        if command -v apt-get &>/dev/null; then
            info "尝试安装: ${missing[*]}"
            apt-get update -qq && apt-get install -y -qq "${missing[@]}" || true
        fi
    fi
}

#============================== 系统监控 ==============================
start_monitor() {
    local csv_file="$1"
    mkdir -p "$(dirname "$csv_file")"

    # CSV 表头
    echo "timestamp,cpu_usage_pct,mem_usage_pct,disk_read_mbs,disk_write_mbs,net_in_kbs,net_out_kbs,gpu_util_pct,gpu_temp_c,gpu_power_w" > "$csv_file"

    (
        while true; do
            local ts
            ts=$(date '+%Y-%m-%d %H:%M:%S')

            # CPU 使用率
            local cpu
            cpu=$(ps -A -o %cpu 2>/dev/null | awk '{s+=$1} END {printf "%.1f", s}')

            # 内存使用率
            local mem
            if command -v free &>/dev/null; then
                mem=$(free | awk '/Mem/ {printf "%.1f", $3/$2*100}')
            elif [[ "$(uname -s)" == "Darwin" ]]; then
                mem=$(vm_stat | awk '/Pages active/ {a=$3} /Pages wired/ {w=$4} /Pages occupied/ {o=$3} END {printf "%.1f", (a+w+o)/999999*100}')
            else
                mem="N/A"
            fi

            # 磁盘 IO
            local disk_r="N/A" disk_w="N/A"
            if command -v iostat &>/dev/null; then
                if [[ "$(uname -s)" == "Darwin" ]]; then
                    local io
                    io=$(iostat -d 1 2 2>/dev/null | tail -1)
                    disk_r=$(echo "$io" | awk '{print $4}')
                    disk_w=$(echo "$io" | awk '{print $5}')
                else
                    local io
                    io=$(iostat -d 1 2 2>/dev/null | tail -1)
                    disk_r=$(echo "$io" | awk '{print $3}')
                    disk_w=$(echo "$io" | awk '{print $4}')
                fi
            fi

            # 网络流量
            local net_in="N/A" net_out="N/A"
            if [[ -f /proc/net/dev ]]; then
                net_in=$(awk '{sum+=$2} END {printf "%.1f", sum/1024}' /proc/net/dev)
                net_out=$(awk '{sum+=$10} END {printf "%.1f", sum/1024}' /proc/net/dev)
            elif command -v netstat &>/dev/null; then
                net_in=$(netstat -ib 2>/dev/null | awk 'NR>1 {sum_in+=$7} END {print sum_in/1024}')
                net_out=$(netstat -ib 2>/dev/null | awk 'NR>1 {sum_out+=$10} END {print sum_out/1024}')
            fi

            # GPU 信息
            local gpu_util="N/A" gpu_temp="N/A" gpu_power="N/A"
            if command -v nvidia-smi &>/dev/null; then
                gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {if(NR>0) printf "%.1f", s/NR; else print "N/A"}')
                gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | sort -nr | head -1)
                gpu_power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | awk '{s+=$1} END {if(NR>0) printf "%.1f", s/NR; else print "N/A"}')
            fi

            echo "${ts},${cpu},${mem},${disk_r},${disk_w},${net_in},${net_out},${gpu_util},${gpu_temp},${gpu_power}" >> "$csv_file"
            sleep "$MONITOR_INTERVAL"
        done
    ) &
    MONITOR_PID=$!
    info "系统监控已启动 (PID: $MONITOR_PID) -> $(basename "$csv_file")"
}

stop_monitor() {
    if [ -n "$MONITOR_PID" ]; then
        kill "$MONITOR_PID" 2>/dev/null || true
        MONITOR_PID=""
    fi
    # 清理残留压力进程
    pkill -f "stress-ng" 2>/dev/null || true
    pkill -f "fio" 2>/dev/null || true
    pkill -f "iperf3" 2>/dev/null || true
}

#============================== GPU 压力测试 ==============================
stress_gpu() {
    header "GPU 压力测试"
    local log_file="${LOG_DIR}/gpu/stress_gpu.log"
    mkdir -p "${LOG_DIR}/gpu"
    local duration_sec=$((DURATION * 60))

    start_monitor "${LOG_DIR}/gpu/monitor.csv"

    local gpu_count=0
    command -v nvidia-smi &>/dev/null && gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)

    if [ "$gpu_count" -gt 0 ]; then
        info "检测到 $gpu_count 个 GPU，启动 stress-ng CUDA 压力测试 (${DURATION}分钟)"
        stress-ng --cuda "$gpu_count" -t "${DURATION}m" --metrics-brief 2>&1 | tee "$log_file"
        pass "GPU stress-ng CUDA 测试完成"
    else
        warn "未检测到 NVIDIA GPU，跳过 GPU 压力测试"
        echo "GPU 不可用，跳过" > "$log_file"
    fi

    stop_monitor
    analyze_log "${LOG_DIR}/gpu/monitor.csv" "GPU"
}

#============================== CPU 压力测试 ==============================
stress_cpu() {
    header "CPU 压力测试"
    local log_file="${LOG_DIR}/cpu/stress_cpu.log"
    mkdir -p "${LOG_DIR}/cpu"
    local duration_sec=$((DURATION * 60))

    start_monitor "${LOG_DIR}/cpu/monitor.csv"

    local cpu_count
    cpu_count=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    info "检测到 $cpu_count 个 CPU 核心，启动压力测试 (${DURATION}分钟)"

    # CPU 压力: FFT + 矩阵 + 整数运算
    stress-ng --cpu "$cpu_count" --cpu-method fft --cpu-method matrixprod \
              --cpu-method int64 --metrics-brief -t "${DURATION}m" 2>&1 | tee "$log_file"

    pass "CPU 压力测试完成"

    stop_monitor
    analyze_log "${LOG_DIR}/cpu/monitor.csv" "CPU"
}

#============================== 内存压力测试 ==============================
stress_memory() {
    header "内存压力测试"
    local log_file="${LOG_DIR}/mem/stress_mem.log"
    mkdir -p "${LOG_DIR}/mem"
    local duration_sec=$((DURATION * 60))

    start_monitor "${LOG_DIR}/mem/monitor.csv"

    local mem_total=0
    if command -v free &>/dev/null; then
        mem_total=$(free -g | awk '/Mem:/ {print int($2*0.8)}')
    fi
    [ "$mem_total" -lt 1 ] && mem_total=4
    info "内存压力: ${mem_total}GB（总内存的 80%）, 持续时间 ${DURATION}分钟"

    # 内存带宽 + 压力
    stress-ng --vm "$(nproc)" --vm-bytes "${mem_total}G" --vm-method all \
              --metrics-brief -t "${DURATION}m" 2>&1 | tee "$log_file"

    pass "内存压力测试完成"

    stop_monitor
    analyze_log "${LOG_DIR}/mem/monitor.csv" "MEMORY"
}

#============================== 硬盘压力测试 ==============================
stress_disk() {
    header "硬盘压力测试"
    local log_file="${LOG_DIR}/disk/stress_disk.log"
    mkdir -p "${LOG_DIR}/disk"
    local duration_sec=$((DURATION * 60))

    start_monitor "${LOG_DIR}/disk/monitor.csv"

    info "启动 fio 顺序/随机读写测试 (${DURATION}分钟)"
    # 顺序读
    fio --name=seq_read --rw=read --bs=1M --size=4G --runtime="${DURATION}m" \
        --time_based --output="${LOG_DIR}/disk/fio_seq_read.log" 2>&1 || true
    # 顺序写
    fio --name=seq_write --rw=write --bs=1M --size=4G --runtime="${DURATION}m" \
        --time_based --output="${LOG_DIR}/disk/fio_seq_write.log" 2>&1 || true
    # 随机读写
    fio --name=rand_rw --rw=randrw --bs=4k --size=4G --runtime="${DURATION}m" \
        --iodepth=64 --ioengine=libaio --direct=1 --group_reporting \
        --time_based --output="${LOG_DIR}/disk/fio_rand_rw.log" 2>&1 || true

    # 汇总 IOPS / 带宽
    {
        echo "===== 硬盘测试汇总 ====="
        for f in "${LOG_DIR}/disk"/fio_*.log; do
            [ -f "$f" ] || continue
            echo "--- $(basename "$f") ---"
            grep -E "IOPS|BW=|WRITE=|READ:" "$f" 2>/dev/null || true
            echo ""
        done
    } > "$log_file"

    pass "硬盘压力测试完成"
    stop_monitor
    analyze_log "${LOG_DIR}/disk/monitor.csv" "DISK"
}

#============================== 网卡压力测试 ==============================
stress_net() {
    header "网卡压力测试"
    local log_file="${LOG_DIR}/net/stress_net.log"
    mkdir -p "${LOG_DIR}/net"
    local duration_sec=$((DURATION * 60))

    start_monitor "${LOG_DIR}/net/monitor.csv"

    # iperf3 需要服务端; 自测用本地回环
    if command -v iperf3 &>/dev/null; then
        info "启动 iperf3 本地回环测试 (${DURATION}分钟)"

        # 启动服务端
        iperf3 -s -D 2>/dev/null || true
        sleep 1

        # TCP 测试（双向）
        iperf3 -c 127.0.0.1 -t "$duration_sec" -P 4 --json 2>/dev/null \
            > "${LOG_DIR}/net/iperf_tcp.json" || true
        iperf3 -c 127.0.0.1 -t "$duration_sec" -P 4 -R --json 2>/dev/null \
            > "${LOG_DIR}/net/iperf_tcp_rev.json" || true

        # UDP 测试
        iperf3 -c 127.0.0.1 -t "$duration_sec" -u -b 0 --json 2>/dev/null \
            > "${LOG_DIR}/net/iperf_udp.json" || true

        # 汇总
        {
            echo "===== 网卡测试汇总 ====="
            echo "--- TCP 下行 ---"
            python3 -c "import json; d=json.load(open('${LOG_DIR}/net/iperf_tcp.json')); print(f'带宽: {d[\"end\"][\"sum_received\"][\"bits_per_second\"]/1e9:.2f} Gbps')" 2>/dev/null || true
            echo "--- TCP 上行 ---"
            python3 -c "import json; d=json.load(open('${LOG_DIR}/net/iperf_tcp_rev.json')); print(f'带宽: {d[\"end\"][\"sum_received\"][\"bits_per_second\"]/1e9:.2f} Gbps')" 2>/dev/null || true
            echo "--- UDP ---"
            python3 -c "import json; d=json.load(open('${LOG_DIR}/net/iperf_udp.json')); print(f'带宽: {d[\"end\"][\"sum\"][\"bits_per_second\"]/1e9:.2f} Gbps, 丢包: {d[\"end\"][\"sum\"][\"lost_percent\"]:.1f}%')" 2>/dev/null || true
        } > "$log_file"

        # 关闭服务端
        pkill -f "iperf3 -s" 2>/dev/null || true
        pass "网卡压力测试完成"
    else
        warn "iperf3 未安装，网卡测试跳过"
        echo "iperf3 不可用，跳过" > "$log_file"
    fi

    stop_monitor
    analyze_log "${LOG_DIR}/net/monitor.csv" "NET"
}

#============================== 监控数据分析 ==============================
analyze_log() {
    local csv="$1"
    local label="$2"
    if [ ! -f "$csv" ]; then
        warn "${label}: 监控数据文件不存在"
        return
    fi
    local lines
    lines=$(wc -l < "$csv")
    [ "$lines" -le 1 ] && return

    local summary="${LOG_DIR}/summary_${label}.log"
    {
        echo "===== ${label} 监控摘要 ====="
        echo "采样点数: $((lines - 1))"
        echo ""
        # CPU
        awk -F',' 'NR>1 && $2!="N/A" {s+=$2; c++} END {if(c>0) printf "平均 CPU 使用率: %.1f%%\n", s/c}' "$csv"
        # 内存
        awk -F',' 'NR>1 && $3!="N/A" {s+=$3; c++} END {if(c>0) printf "平均 内存使用率: %.1f%%\n", s/c}' "$csv"
        # GPU
        awk -F',' 'NR>1 && $8!="N/A" {s+=$8; c++} END {if(c>0) printf "平均 GPU 利用率: %.1f%%\n", s/c}' "$csv"
        awk -F',' 'NR>1 && $9!="N/A" {if($9>m)m=$9} END {if(m>0) printf "最高 GPU 温度: %.0f°C\n", m}' "$csv"
        awk -F',' 'NR>1 && $10!="N/A" {if($10>m)m=$10} END {if(m>0) printf "最高 GPU 功耗: %.1fW\n", m}' "$csv"
    } > "$summary"
    info "${label} 摘要已保存: $(basename "$summary")"
}

#============================== 报告生成 ==============================
generate_report() {
    header "生成测试报告"
    local report="${LOG_DIR}/stress_report.txt"

    {
        echo "================================================================================"
        echo "                      整机压力测试报告"
        echo "================================================================================"
        echo "测试时间: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "每部件时长: ${DURATION} 分钟"
        echo "日志目录: ${LOG_DIR}"
        echo ""
        echo "--------------------------------------------------------------------------------"
        echo "测试部件与结果"
        echo "--------------------------------------------------------------------------------"
    } > "$report"

    for comp in "${COMPONENTS[@]}"; do
        local summary="${LOG_DIR}/summary_${comp^^}.log"
        if [ -f "$summary" ]; then
            echo "" >> "$report"
            cat "$summary" >> "$report"
        fi
    done

    echo "" >> "$report"
    echo "================================================================================" >> "$report"
    echo "报告生成完毕" >> "$report"
    echo "================================================================================" >> "$report"

    cat "$report"
    info "报告已保存: $report"

    # 生成 CSV 汇总
    local csv_summary="${LOG_DIR}/stress_summary.csv"
    echo "部件,平均CPU(%),平均内存(%),平均GPU利用率(%),最高GPU温度(°C),最高GPU功耗(W)" > "$csv_summary"
    for comp in "${COMPONENTS[@]}"; do
        local csv_file="${LOG_DIR}/${comp}/monitor.csv"
        if [ -f "$csv_file" ] && [ "$(wc -l < "$csv_file")" -gt 1 ]; then
            local avg_cpu avg_mem avg_gpu max_temp max_power
            avg_cpu=$(awk -F',' 'NR>1 && $2!="N/A" {s+=$2; c++} END {if(c>0) printf "%.1f", s/c; else print "N/A"}' "$csv_file")
            avg_mem=$(awk -F',' 'NR>1 && $3!="N/A" {s+=$3; c++} END {if(c>0) printf "%.1f", s/c; else print "N/A"}' "$csv_file")
            avg_gpu=$(awk -F',' 'NR>1 && $8!="N/A" {s+=$8; c++} END {if(c>0) printf "%.1f", s/c; else print "N/A"}' "$csv_file")
            max_temp=$(awk -F',' 'NR>1 && $9!="N/A" {if($9>m)m=$9} END {printf "%.0f", m}' "$csv_file")
            max_power=$(awk -F',' 'NR>1 && $10!="N/A" {if($10>m)m=$10} END {printf "%.1f", m}' "$csv_file")
            echo "${comp},${avg_cpu},${avg_mem},${avg_gpu},${max_temp},${max_power}" >> "$csv_summary"
        fi
    done
    info "数据汇总已保存: $(basename "$csv_summary")"
}

#============================== 进度展示 ==============================
show_progress() {
    local current=$1 total=$2 name=$3
    echo ""
    echo "================================================================================"
    printf "进度: [%d/%d] %d%%  |  当前: %s\n" "$current" "$total" "$((current * 100 / total))" "$name"
    echo "================================================================================"
}

#============================== 清理 ==============================
cleanup() {
    stop_monitor
    pkill -f "stress-ng|fio|iperf3" 2>/dev/null || true
    exit 0
}
trap cleanup EXIT INT TERM

#============================== 帮助 ==============================
show_help() {
    cat <<EOF
整机压力测试脚本 — 单元测试模式

用法: $0 [选项]

选项:
    -d, --duration <分钟>    每部件测试时长（默认: 10）
    -c, --components <列表>   测试部件，逗号分隔（默认: gpu,cpu,mem,disk,net）
    -l, --list               列出可用部件
    -h, --help               显示此帮助

部件列表:
    gpu   GPU  压力测试（stress-ng CUDA / nvidia-smi 监控）
    cpu   CPU  压力测试（FFT + 矩阵 + 整数运算）
    mem   内存 压力测试（VM stress + 带宽测试）
    disk  硬盘 压力测试（fio 顺序/随机读写）
    net   网卡 压力测试（iperf3 本地回环）

示例:
    $0                         # 全量测试（每部件 10 分钟）
    $0 -d 30                   # 每部件 30 分钟
    $0 -c cpu,mem,disk         # 仅测试 CPU + 内存 + 硬盘
    $0 -c gpu -d 60            # 仅 GPU 测试 60 分钟

环境变量:
    DURATION                  每部件测试时长（分钟），默认 10
    MONITOR_INTERVAL          监控采样间隔（秒），默认 2
EOF
}

list_components() {
    echo "可用部件:"
    for c in "${ALL_COMPONENTS[@]}"; do
        echo "  $c"
    done
}

#============================== 主函数 ==============================
main() {
    # 解析参数
    COMPONENTS=("${ALL_COMPONENTS[@]}")
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -d|--duration)
                DURATION="$2"; shift 2 ;;
            -c|--components)
                IFS=',' read -ra COMPONENTS <<< "$2"; shift 2 ;;
            -l|--list)
                list_components; exit 0 ;;
            -h|--help)
                show_help; exit 0 ;;
            *)
                echo "未知参数: $1"
                show_help
                exit 1 ;;
        esac
    done

    echo "================================================================================"
    echo "                      整机压力测试"
    echo "================================================================================"
    echo "每部件测试时长: ${DURATION} 分钟"
    echo "测试部件: ${COMPONENTS[*]}"
    echo "日志目录: ${LOG_DIR}"
    echo "================================================================================"

    # 创建目录
    for comp in "${COMPONENTS[@]}"; do
        mkdir -p "${LOG_DIR}/${comp}"
    done

    # 检查工具
    check_prereqs

    local total=${#COMPONENTS[@]}
    local current=0

    for comp in "${COMPONENTS[@]}"; do
        current=$((current + 1))
        show_progress "$current" "$total" "$comp"
        case "$comp" in
            gpu)  stress_gpu ;;
            cpu)  stress_cpu ;;
            mem)  stress_memory ;;
            disk) stress_disk ;;
            net)  stress_net ;;
            *)    warn "未知部件: $comp，跳过" ;;
        esac
    done

    generate_report

    echo ""
    echo "================================================================================"
    echo "全量测试完成！日志目录: ${LOG_DIR}"
    echo "================================================================================"
}

main "$@"
