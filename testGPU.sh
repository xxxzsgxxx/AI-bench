#!/bin/bash
# 这个脚本是参考5D测试关于GPU的测试用例，修改了部分代码，增加了7个测试阶段。在运行之前增加了系统信息资源采集，burning时候增加了系统状态采集。
# 
# by scott, 20250510

# 模块化主控脚本
set -eo pipefail

# 新增CUDA Sample路径定义
CUDA_SAMPLE_DIR="/root/cuda-samples/build/Samples"  # 默认安装路径
    if [ ! -d "${CUDA_SAMPLE_DIR}" ]; then
        echo "错误：CUDA Sample目录不存在 ${CUDA_SAMPLE_DIR}" >&2
        exit 1
    fi

# 检测必要工具存在性
    command -v ipmitool >/dev/null 2>&1 || { echo >&2 "需要安装ipmitool工具"; return 1; }
    command -v /root/gpu-burn/gpu_burn >/dev/null 2>&1 || { echo >&2 "需要安装gpu_burn测试工具"; return 1; }
    command -v hwinfo >/dev/null 2>&1 || { echo >&2 "需要安装hwinfo硬件检测工具"; return 1; }
    command -v dmidecode >/dev/null 2>&1 || { echo >&2 "需要安装dmidecode工具"; return 1; }
    command -v smartctl >/dev/null 2>&1 || { echo >&2 "需要安装smartctl工具"; return 1; }
    command -v nvme >/dev/null 2>&1 || { echo >&2 "需要安装nvme工具"; return 1; }
    command -v nvtop >/dev/null 2>&1 || { echo >&2 "需要安装nvtop工具"; return 1; }

# 初始化日志系统
init_logging() {
    LOG_BASE="./TESTLOG_$(date +%Y%m%d_%H%M)"
    mkdir -p "${LOG_BASE}"/{sysinfo,stage_logs,monitor_logs}
}


test01() {
    echo "执行阶段01：CUDA P2P Bandwidth latency test 测试..."
    ${CUDA_SAMPLE_DIR}/5_Domain_Specific/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest | tee ${LOG_BASE}/PERF_GPU00001.log 2>&1
}

# 新增7个测试阶段函数
test02() {
    echo "执行阶段02：H2D单卡测试..."
    ${CUDA_SAMPLE_DIR}/1_Utilities/bandwidthTest/bandwidthTest --mode=range --mode=range --start=1024000 --end=64000000  \
    	--increment=10240 --htod  --device=all | tee "${LOG_BASE}/PERF_GPU00002_H2D.log"
}

# 在脚本开头添加GPU数量检测函数
detect_gpu_count() {
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    echo "检测到 ${GPU_COUNT} 个GPU设备"
    if [ $GPU_COUNT -eq 0 ]; then
        echo "错误：未检测到GPU设备，无法执行测试" >&2
        exit 1
    fi
}

# 修改test03函数
test03() {
    echo "执行阶段03：H2D多卡并行测试..."
    local pids=()
    # 使用动态检测的GPU数量
    for device_id in $(seq 0 $((GPU_COUNT-1))); do
        # H2D测试后台执行，输出重定向到日志文件
        ${CUDA_SAMPLE_DIR}/1_Utilities/bandwidthTest/bandwidthTest --mode=range --start=1024000 --end=64000000 \
            --increment=10240 --htod --device=$device_id | tee ${LOG_BASE}/PERF_GPU00003_gpu${device_id}_H2D.log 2>&1 &
        pids+=($!)
    done
    # 等待所有测试完成
    wait ${pids[@]}
}

test04() {
    echo "执行阶段04：D2H单卡测试..."
    ${CUDA_SAMPLE_DIR}/1_Utilities/bandwidthTest/bandwidthTest --mode=range --start=1024000 --end=64000000  \
        --increment=10240 --dtoh --device=all | tee "${LOG_BASE}/PERF_GPU00004_D2H.log"
    }

test05() {
    echo "执行阶段05：D2H多卡并行测试"
    local pids=()
    for device_id in $(seq 0 $((GPU_COUNT-1))); do
        ${CUDA_SAMPLE_DIR}/1_Utilities/bandwidthTest/bandwidthTest --mode=range --start=1024000 --end=64000000 \
            --increment=10240 --dtoh --device=$device_id | tee ${LOG_BASE}/PERF_GPU00005_gpu${device_id}_D2H.log 2>&1 &
        pids+=($!)
    done
    # 等待所有测试完成
    wait ${pids[@]}
}


test06() {
    echo "执行阶段06：D2D测试... [$(date +'%Y-%m-%d %H:%M:%S')]"
    ${CUDA_SAMPLE_DIR}/1_Utilities/bandwidthTest/bandwidthTest --mode=range --start=1024000 --end=64000000 \
        --increment=10240 --dtod --device=all | tee ${LOG_BASE}/PERF_GPU00006.log 2>&1 
}

test07() {
    echo "执行阶段07：H2D_D2H单卡并行测试..."

    for device_id in $(seq 0 $((GPU_COUNT-1))); do
        # 启动当前设备双测试
        ${CUDA_SAMPLE_DIR}/1_Utilities/bandwidthTest/bandwidthTest --mode=range --start=1024000 --end=64000000 \
            --increment=10240 --htod --device=$device_id | tee ${LOG_BASE}/PERF_GPU00007_gpu${device_id}_htod.log 2>&1 &
        htod_pid=$!

        ${CUDA_SAMPLE_DIR}/1_Utilities/bandwidthTest/bandwidthTest --mode=range --start=1024000 --end=64000000 \
            --increment=10240 --dtoh --device=$device_id | tee ${LOG_BASE}/PERF_GPU00007_gpu${device_id}_dtoh.log 2>&1 &
        dtoh_pid=$!

        # 等待本设备测试完成
        wait $htod_pid $dtoh_pid
        echo "GPU $device_id 测试完成"
    done
}

test08() {
    echo "执行阶段08：H2D_D2H多卡并行测试..."
    local pids=()
    for device_id in $(seq 0 $((GPU_COUNT-1))); do
        # H2D测试后台执行，输出重定向到日志文件
        ${CUDA_SAMPLE_DIR}/1_Utilities/bandwidthTest/bandwidthTest --mode=range --start=1024000 --end=64000000 \
            --increment=10240 --htod --device=$device_id | tee ${LOG_BASE}/PERF_GPU00008_gpu${device_id}_htod.log 2>&1 &
        pids+=($!)

        # D2H测试后台执行，输出重定向到日志文件
        ${CUDA_SAMPLE_DIR}/1_Utilities/bandwidthTest/bandwidthTest --mode=range --start=1024000 --end=64000000 \
            --increment=10240 --dtoh --device=$device_id | tee ${LOG_BASE}/PERF_GPU00008_gpu${device_id}_dtoh.log 2>&1 &
        pids+=($!)
    done
    # 等待所有测试完成
    wait ${pids[@]}
}


test09() {
    echo "执行阶段09：GPU压力测试 [$(date +'%Y-%m-%d %H:%M:%S')]"
    # 启动gpu_burn一小时测试
    cd ~/gpu-burn/; ./gpu_burn 3600 | tee ${LOG_BASE}/GPU-GURN-3600.log 2>&1 &
    GPU_BURN_PID=$!

    # 30分钟后启动高负载sensor采集
    (
        sleep 1800  # 等待30分钟
        echo "=== 启动高负载sensor采集 [$(date +'%Y-%m-%d %H:%M:%S')] ==="
        while true; do
            ipmitool sensor list >> ${LOG_BASE}/monitor_logs/highload_sensors.log
            sleep 60  # 每5分钟采集一次
        done
    ) &
    IPMI_PGID=$!
    disown -h $IPMI_PGID  # 防止被清理函数误杀

    wait $GPU_BURN_PID
    kill $IPMI_PID
}



# 执行环境初始化
execute_init() {
    echo "=== 系统环境初始化 ==="
    detect_gpu_count
}

# 执行系统信息采集
execute_sysinfo() {
    echo "正在执行系统信息采集..."
# 1. 统一硬件检测调用方式
lshw  2>&1 | tee -a ${LOG_BASE}/sysinfo/system_hardware.log
dmidecode -t bios 2>&1 | tee -a ${LOG_BASE}/sysinfo/bios_info.log
dmesg > ${LOG_BASE}/sysinfo/dmesg.log
lspci -vt > ${LOG_BASE}/sysinfo/pci_tree.log
lspci -vvv > ${LOG_BASE}/sysinfo/pci_info.log
lscpu > ${LOG_BASE}/sysinfo/cpu_info.log
lstopo > ${LOG_BASE}/sysinfo/lstopo.log
free -h > ${LOG_BASE}/sysinfo/memory_info.log
df -h > ${LOG_BASE}/sysinfo/disk_space.log

# 1.2 获取NVMe硬盘信息
smartctl -A /dev/nvme0n1 > ${LOG_BASE}/sysinfo/nvme0_smart.log 2>&1 || true
smartctl -A /dev/nvme1n1 > ${LOG_BASE}/sysinfo/nvme1_smart.log 2>&1 || true
smartctl -A /dev/nvme2n1 > ${LOG_BASE}/sysinfo/nvme3_smart.log 2>&1 || true
nvme smart-log /dev/nvme0n1 > ${LOG_BASE}/sysinfo/nvme_smart_log.log 
nvme id-ctrl /dev/nvme0n1 > ${LOG_BASE}/sysinfo/nvme_id_ctrl.log
nvme id-ns /dev/nvme0n1 > ${LOG_BASE}/sysinfo/nvme_id_ns.log

/root/SCELNX_64 /o /s  ${LOG_BASE}/sysinfo/BIOS_setup.log
ipmitool fru print > ${LOG_BASE}/sysinfo/fru_info.log
ipmitool lan print  > ${LOG_BASE}/sysinfo/lan_info.log
ipmitool sensor list > ${LOG_BASE}/sysinfo/sensor_info.log
nvidia-smi > ${LOG_BASE}/sysinfo/nvidia_smi.log
nvidia-smi -q > ${LOG_BASE}/sysinfo/nvidia_smi_query.log
nvidia-smi -L > ${LOG_BASE}/sysinfo/nvidia_smi_list.log
nvidia-smi topo -m > ${LOG_BASE}/sysinfo/nvidia_smi_topo.log
nvidia-smi topo -p2p p > ${LOG_BASE}/sysinfo/nvidia_smi_topo_p2p.log
/usr/local/cuda/extras/demo_suite/deviceQuery > ${LOG_BASE}/sysinfo/cuda_deviceQuery.log || true
/usr/local/cuda/extras/demo_suite/busGrind > ${LOG_BASE}/sysinfo/cuda_busGrind.log  || true
/usr/local/cuda/extras/demo_suite/vectorAdd > ${LOG_BASE}/sysinfo/cuda_vectorAdd.log || true


# 2. 系统服务信息
cat /etc/os-release > ${LOG_BASE}/sysinfo/os_release.log
cat /proc/version > ${LOG_BASE}/sysinfo/kernel_version.log
cat /proc/driver/nvidia/version > ${LOG_BASE}/sysinfo/nvidia_driver_version.log
cat /proc/cpuinfo > ${LOG_BASE}/sysinfo/cpu_info.log
cat /proc/meminfo > ${LOG_BASE}/sysinfo/memory_info.log
cat /proc/interrupts > ${LOG_BASE}/sysinfo/interrupts.log
cat /proc/net/dev > ${LOG_BASE}/sysinfo/network_devices.log
cat /proc/net/route > ${LOG_BASE}/sysinfo/network_routes.log
cat /proc/net/arp > ${LOG_BASE}/sysinfo/network_arp.log
cat /proc/net/if_inet6 > ${LOG_BASE}/sysinfo/network_ipv6.log
lsmod > ${LOG_BASE}/sysinfo/modules.log
uname -a > ${LOG_BASE}/sysinfo/kernel_info.log

systemctl list-units > ${LOG_BASE}/sysinfo/services.log
journalctl -k --since "1 hour ago" > ${LOG_BASE}/sysinfo/kernel_journal.log

# 3. 进程/环境信息
ps auxf > ${LOG_BASE}/sysinfo/process_list.log
env > ${LOG_BASE}/sysinfo/environment_vars.log

# 4. 网络/存储增强
lsblk  > ${LOG_BASE}/sysinfo/block_devices.log
mdadm --detail /dev/md0 > ${LOG_BASE}/sysinfo/raid_status.log 2>&1 || echo "RAID状态获取失败"

# 5. 性能基准参考
sysctl -a > ${LOG_BASE}/sysinfo/ysctl_params.log
uptime >> ${LOG_BASE}/sysinfo/system_load.log

}

# 生成测试报告
generate_report() {
    echo "\n=== 生成最终测试报告 ==="
    #../report_generator/report_generator.sh "${LOG_BASE}"
}




# 修改监控函数（降低采集频率）
start_monitoring() {
    mkdir -p "${LOG_BASE}/monitor_logs"
    local gpu_count=$(nvidia-smi -L | wc -l)
    
    # 系统资源监控（每10秒采样）
    {
        echo "时间戳,CPU使用率(%),内存使用率(%),GPU平均使用率(%),GPU最高温度(℃)"
        while true; do
            timestamp=$(date +%'Y-%m-%d %H:%M:%S')
            cpu_usage=$(ps -A -o %cpu | awk '{s+=$1} END {print s}')
            mem_usage=$(vmstat | awk '/free/ {print $3}' | tr -d '.')
            
            gpu_count=$(nvidia-smi -L | wc -l)
            if [ $gpu_count -gt 0 ]; then
                gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{sum+=$1} END {printf "%.1f", sum/NR}')
                gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -nr | head -1)
            else
                gpu_util="N/A"
                gpu_temp="N/A"
            fi

            echo "${timestamp},${cpu_usage},${mem_usage},${gpu_util},${gpu_temp}"
            sleep 10  # 降低到每10s采集一次
        done
    } > "${LOG_BASE}/monitor_logs/system_monitor.csv" &
    MONITOR_PID=$!
}

stop_monitoring() {
    # 终止所有相关进程
    if [ -n "$MONITOR_PID" ]; then
        kill $MONITOR_PID 2>/dev/null || true
    fi
    
    pkill -f "nvidia-smi --query-gpu" || true
    
    # 修复方案（精确匹配进程）
    pkill -f "nvidia-smi --query-gpu=utilization.gpu" || true
    pkill -f "system_monitor.csv" || true
    pkill -f "gpu_detail.csv" || true
    pkill -f "ipmitool sensor list" || true
    pkill -f "start_monitoring" || true
    
    # 确保所有监控子进程都被终止
    for pid in $(jobs -p); do
        kill $pid 2>/dev/null || true
    done
    
    # 修复awk除零错误
    gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{sum+=$1} END {if(NR>0) printf "%.1f", sum/NR; else print "N/A"}')
}


execute_benchmark() {
    echo -e "\n=== 启动七阶段性能测试 [$(date +'%Y-%m-%d %H:%M:%S')] ==="
    
    local stages=(test01 test02 test03 test04 test05 test06 test07 test08 test09)
    
    for stage in "${stages[@]}"; do
        local stage_num=$(printf "%02d" $((10#${stage#test})))
        echo "=== 阶段 ${stage_num} 启动 [$(date +'%Y-%m-%d %H:%M:%S')] ===" | tee "${LOG_BASE}/stage_logs/stage_${stage_num}.log"
        
        start_monitoring
        if { $stage || exit $?; } | tee -a "${LOG_BASE}/stage_logs/stage_${stage_num}.log"; then
            echo "阶段 ${stage_num} 完成 [$(date +'%Y-%m-%d %H:%M:%S')]" | tee -a "${LOG_BASE}/stage_logs/stage_${stage_num}.log"
        else
            echo "阶段 ${stage_num} 失败！错误码 $? [$(date +'%Y-%m-%d %H:%M:%S')]" | tee -a "${LOG_BASE}/stage_logs/stage_${stage_num}.log"
            exit 1
        fi
        stop_monitoring
    done
    
    echo -e "\n=== 所有测试阶段完成 [$(date +'%Y-%m-%d %H:%M:%S')] ==="
}

main() {
    local start_time=$(date +%s)
    # 定义退出清理函数
    cleanup() {
        echo -e "\n捕获中断信号，清理进程..."
        stop_monitoring
        # 原cleanup函数（第321行）
        pkill -P $$ 
        
        # 修复方案（使用进程组清理）
        pkill -TERM -P $$ 2>/dev/null || true
        kill -- -$$ 2>/dev/null || true
        exit 1
    }
    
    # 注册信号捕获
    trap cleanup SIGINT SIGTERM
    
    parse_arguments "$@" 
    init_logging
    
    # 原参数判断逻辑保持不变
    if [[ $FULL_MODE ]]; then
        execute_init
        execute_sysinfo
        execute_benchmark
        generate_report
    else
        [[ $INIT_MODE ]] && execute_init
        [[ $SYSINFO_MODE ]] && execute_sysinfo
        [[ $BENCH_MODE ]] && execute_benchmark
        [[ $REPORT_MODE ]] && generate_report
    fi

    echo -e "\n操作完成！日志目录: ${LOG_BASE}"
    
    # 计算总耗时
    local end_time=$(date +%s)
    local total_seconds=$((end_time - start_time))
    local hours=$((total_seconds / 3600))
    local minutes=$(( (total_seconds % 3600) / 60 ))
    local seconds=$((total_seconds % 60))
    
    echo -e "\n=== 总运行时间：${hours}小时 ${minutes}分 ${seconds}秒 ==="
    
    # 确保所有进程在退出前被终止
    cleanup_all_processes
    exit 0
}

# 添加一个新的清理函数，确保所有进程在退出前被终止
cleanup_all_processes() {
    echo "清理所有测试进程..."
    stop_monitoring
    
    # 终止所有可能的残留进程
    pkill -f "bandwidthTest" || true
    pkill -f "p2pBandwidthLatencyTest" || true
    pkill -f "gpu_burn" || true
    
    # 等待所有后台作业完成
    wait 2>/dev/null || true
}

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --full) FULL_MODE=true ;;
            --init) INIT_MODE=true ;;
            --sysinfo) SYSINFO_MODE=true ;;
            --benchmark) BENCH_MODE=true ;;
            --report) REPORT_MODE=true ;;
        esac
        shift
    done

    # 修复方案（添加无效参数处理）
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --full) FULL_MODE=true ;;
            --*) 
                echo "无效参数: $1"
                exit 1
                ;;
        esac
        shift
    done
}


main "$@"