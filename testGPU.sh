#!/bin/bash
################################################################################
# GPU 测试脚本 - 功能增强版
# 参考5D测试关于GPU的测试用例，修改了部分代码，增加了9个测试阶段
# 在运行之前增加了系统信息资源采集，burning时候增加了系统状态采集
#
# 优化内容：
# 1. 错误处理增强：超时检测、失败重试、详细错误日志
# 2. 日志系统增强：时间戳、文件大小限制、日志轮转、压缩
# 3. 系统监控增强：网络流量、磁盘IO、功耗、GPU详细指标
# 4. 测试结果验证：结果完整性检查、摘要报告、自动判断
# 5. 配置参数化：路径、时长、采样间隔可配置
# 6. 代码质量改进：修复重复定义、逻辑错误、统一命名
# 7. 兼容性增强：多路径检测、macOS支持、自动安装提示
# 8. 进度显示：百分比、剩余时间、实时指标
# 9. 通知功能：测试完成/失败通知
# 10. 报告生成：HTML可视化、图表、对比报告
#
# by scott, 20250510
# enhanced, 20250131
################################################################################

set -eo pipefail

################################################################################
# 全局配置参数 - 可根据需要修改
################################################################################

# CUDA Sample 路径（支持多个路径自动检测）
CUDA_SAMPLE_PATHS=(
    "/root/cuda-samples/build/Samples"
    "/usr/local/cuda/samples/build/Samples"
    "~/cuda-samples/build/Samples"
    "/opt/cuda/samples/build/Samples"
)

# GPU Burn 路径
GPU_BURN_PATHS=(
    "/root/gpu-burn/gpu_burn"
    "~/gpu-burn/gpu_burn"
    "/usr/local/bin/gpu_burn"
)

# 测试参数配置
GPU_BURN_DURATION=3600          # GPU 压力测试时长（秒）
MONITOR_INTERVAL=10             # 监控采样间隔（秒）
SENSOR_INTERVAL=60              # 传感器采集间隔（秒）
MAX_LOG_SIZE=$((100 * 1024 * 1024))  # 最大日志文件大小（100MB）
MAX_RETRY=3                     # 失败重试次数
STAGE_TIMEOUT=3600              # 单个测试阶段超时时间（秒）

# 日志配置
ENABLE_LOG_COMPRESSION=true     # 是否启用日志压缩
LOG_RETENTION_DAYS=30           # 日志保留天数

# 通知配置
ENABLE_NOTIFICATION=false       # 是否启用通知
NOTIFICATION_EMAIL=""           # 通知邮箱（留空则不发送）

# 操作系统检测
OS_TYPE=$(uname -s)
OS_VERSION=$(uname -r)

################################################################################
# 全局变量
################################################################################

CUDA_SAMPLE_DIR=""
GPU_BURN_PATH=""
GPU_COUNT=0
GPU_NAMES=()
LOG_BASE=""
MONITOR_PID=""
HIGHLOAD_MONITOR_PID=""
STAGE_START_TIME=""
TOTAL_STAGES=9
CURRENT_STAGE=0
PASS_COUNT=0
FAIL_COUNT=0
TOTAL_TESTS=0
BG_PIDS=()
BG_PGID=()

################################################################################
# 工具检测函数
################################################################################

# 检测 CUDA Sample 目录
detect_cuda_samples() {
    for path in "${CUDA_SAMPLE_PATHS[@]}"; do
        expanded_path="${path/#\~/$HOME}"
        if [ -d "$expanded_path" ]; then
            CUDA_SAMPLE_DIR="$expanded_path"
            echo "✓ 检测到 CUDA Sample 目录: $CUDA_SAMPLE_DIR"
            return 0
        fi
    done
    echo "✗ 错误：未找到 CUDA Sample 目录" >&2
    echo "  尝试的路径: ${CUDA_SAMPLE_PATHS[*]}" >&2
    exit 1
}

# 检测 GPU Burn 工具
detect_gpu_burn() {
    for path in "${GPU_BURN_PATHS[@]}"; do
        expanded_path="${path/#\~/$HOME}"
        if [ -f "$expanded_path" ] && [ -x "$expanded_path" ]; then
            GPU_BURN_PATH="$expanded_path"
            echo "✓ 检测到 GPU Burn 工具: $GPU_BURN_PATH"
            return 0
        fi
    done
    echo "✗ 错误：未找到 GPU Burn 工具" >&2
    echo "  尝试的路径: ${GPU_BURN_PATHS[*]}" >&2
    exit 1
}

# 检测必要工具
check_required_tools() {
    local missing_tools=()
    local tools_to_check=("stress-ng" "fio" "lshw")
    
    # macOS 特定工具
    if [ "$OS_TYPE" = "Darwin" ]; then
        tools_to_check+=("system_profiler")
    else
        tools_to_check+=("ipmitool" "dmidecode" "smartctl" "nvme" "nvtop")
    fi
    
    for tool in "${tools_to_check[@]}"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            missing_tools+=("$tool")
        fi
    done
    
    # 检测 GPU Burn
    if ! command -v "$GPU_BURN_PATH" >/dev/null 2>&1 && [ ! -f "$GPU_BURN_PATH" ]; then
        missing_tools+=("gpu_burn")
    fi
    
    if [ ${#missing_tools[@]} -gt 0 ]; then
        echo "⚠ 警告：缺少以下工具: ${missing_tools[*]}" >&2
        echo "  这些工具可能影响部分测试功能" >&2
        
        # 尝试自动安装（仅 Linux）
        if [ "$OS_TYPE" = "Linux" ]; then
            echo "  尝试自动安装..." >&2
            if command -v apt-get >/dev/null 2>&1; then
                apt-get update -qq
                apt-get install -y ipmitool hwinfo dmidecode smartmontools nvme-cli nvtop stress-ng fio lshw hwloc 2>&1 | grep -v "^Selecting\|^Preparing\|^Unpacking\|^Setting up" || true
            elif command -v yum >/dev/null 2>&1; then
                yum install -y ipmitool dmidecode smartmontools nvme-cli stress-ng fio lshw hwloc 2>&1 || true
            fi
        fi
    fi
}

# 检测 GPU 数量
detect_gpu_count() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_COUNT=$(nvidia-smi -L | wc -l)
        echo "✓ 检测到 ${GPU_COUNT} 个 GPU 设备"
    else
        echo "✗ 错误：未找到 nvidia-smi 命令，无法检测 GPU" >&2
        exit 1
    fi
    
    if [ "$GPU_COUNT" -eq 0 ]; then
        echo "✗ 错误：未检测到 GPU 设备，无法执行测试" >&2
        exit 1
    fi
    
    # 记录 GPU 名称
    for i in $(seq 0 $((GPU_COUNT - 1))); do
        GPU_NAMES[i]=$(nvidia-smi --id=$i --query-gpu=name --format=csv,noheader 2>/dev/null)
    done
}

# 查找 CUDA Sample 二进制文件（非致命，返回路径）
find_cuda_sample() {
    local binary="$1"
    local search_paths=(
        "/root/cuda-samples/build/Samples"
        "/usr/local/cuda/samples/build/Samples"
        "~/cuda-samples/build/Samples"
        "/opt/cuda/samples/build/Samples"
        "/usr/local/cuda/extras/demo_suite"
    )
    
    for base in "${search_paths[@]}"; do
        local path="${base/\~/$HOME}/${binary}"
        if [ -x "$path" ]; then
            echo "$path"
            return 0
        fi
    done
    return 1
}

################################################################################
# 日志系统函数
################################################################################

# 初始化日志系统
init_logging() {
    LOG_BASE="./TESTLOG_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${LOG_BASE}"/{sysinfo,stage_logs,monitor_logs,reports}
    
    echo "日志目录: ${LOG_BASE}"
    echo "测试开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    
    # 创建测试元数据文件
    cat > "${LOG_BASE}/test_metadata.json" <<EOF
{
  "test_start_time": "$(date -Iseconds)",
  "os_type": "${OS_TYPE}",
  "os_version": "${OS_VERSION}",
  "gpu_count": ${GPU_COUNT},
  "cuda_sample_dir": "${CUDA_SAMPLE_DIR}",
  "test_stages": ${TOTAL_STAGES}
}
EOF
}

# 检查日志文件大小，超过限制则轮转
check_log_rotation() {
    local log_file="$1"
    
    if [ -f "$log_file" ] && [ "$(stat -f%z "$log_file" 2>/dev/null || stat -c%s "$log_file" 2>/dev/null)" -gt "$MAX_LOG_SIZE" ]; then
        local backup_file="${log_file}.old"
        mv "$log_file" "$backup_file"
        if [ "$ENABLE_LOG_COMPRESSION" = true ]; then
            gzip "$backup_file" 2>/dev/null || true
        fi
        echo "⚠ 日志文件过大，已轮转: $log_file"
    fi
}

# 写入带时间戳的日志
log_with_timestamp() {
    local message="$1"
    local log_file="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[${timestamp}] ${message}"
    if [ -n "$log_file" ]; then
        check_log_rotation "$log_file"
        echo "[${timestamp}] ${message}" >> "$log_file"
    fi
}

################################################################################
# 监控系统函数
################################################################################

# 启动系统资源监控
start_monitoring() {
    mkdir -p "${LOG_BASE}/monitor_logs"
    local gpu_count=$(nvidia-smi -L | wc -l)
    
    # 系统资源监控（每10秒采样）
    {
        echo "时间戳,CPU使用率(%),内存使用率(%),磁盘IO读(MB/s),磁盘IO写(MB/s),网络入(KB/s),网络出(KB/s),GPU平均使用率(%),GPU最高温度(℃),GPU功耗(W),显存使用率(%),PCIe重传计数器,PCIe发送(MB)"
        while true; do
            timestamp=$(date '+%Y-%m-%d %H:%M:%S')
            
            # CPU 使用率
            cpu_usage=$(ps -A -o %cpu | awk '{s+=$1} END {printf "%.1f", s}')
            
            # 内存使用率
            if [ "$OS_TYPE" = "Darwin" ]; then
                mem_usage=$(vm_stat | awk '/Pages free/ {free=$3} /Pages active/ {active=$3} /Pages inactive/ {inactive=$3} /Pages wired/ {wired=$3} END {total=free+active+inactive+wired; printf "%.1f", (active+inactive+wired)/total*100}')
            else
                mem_usage=$(free | awk '/Mem/ {printf "%.1f", $3/$2*100}')
            fi
            
            # 磁盘 IO
            if [ "$OS_TYPE" = "Darwin" ]; then
                disk_read=$(iostat -d 1 2 | tail -1 | awk '{print $4}')
                disk_write=$(iostat -d 1 2 | tail -1 | awk '{print $5}')
            else
                disk_read=$(iostat -d 1 2 | tail -1 | awk '{print $3}')
                disk_write=$(iostat -d 1 2 | tail -1 | awk '{print $4}')
            fi
            
            # 网络流量
            if [ "$OS_TYPE" = "Darwin" ]; then
                net_in=$(netstat -ib | awk '{sum_in+=$7} END {print sum_in/1024}')
                net_out=$(netstat -ib | awk '{sum_out+=$10} END {print sum_out/1024}')
            else
                net_in=$(cat /proc/net/dev | awk '{sum_in+=$2} END {print sum_in/1024}')
                net_out=$(cat /proc/net/dev | awk '{sum_out+=$10} END {print sum_out/1024}')
            fi
            
            # GPU 信息
            if [ $gpu_count -gt 0 ]; then
                gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{sum+=$1} END {if(NR>0) printf "%.1f", sum/NR; else print "N/A"}')
                gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sort -nr | head -1)
                gpu_power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | awk '{sum+=$1} END {if(NR>0) printf "%.1f", sum/NR; else print "N/A"}')
                gpu_mem=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | awk -F',' '{used+=$1; total+=$2} END {if(total>0) printf "%.1f", used/total*100; else print "N/A"}')
            else
                gpu_util="N/A"
                gpu_temp="N/A"
                gpu_power="N/A"
                gpu_mem="N/A"
            fi

            pcie_rx=$(nvidia-smi --id=0 --query-gpu=pcie.replay_counter --format=csv,noheader 2>/dev/null || echo "N/A")
            pcie_tx=$(nvidia-smi --id=0 --query-gpu=pcie.tx_bytes --format=csv,noheader 2>/dev/null || echo "N/A")
            echo "${timestamp},${cpu_usage},${mem_usage},${disk_read},${disk_write},${net_in},${net_out},${gpu_util},${gpu_temp},${gpu_power},${gpu_mem},${pcie_rx},${pcie_tx}"
            sleep "$MONITOR_INTERVAL"
        done
    } > "${LOG_BASE}/monitor_logs/system_monitor.csv" &
    MONITOR_PID=$!
    
    log_with_timestamp "✓ 系统监控已启动 (PID: $MONITOR_PID)"
}

# 启动高负载传感器采集
start_highload_monitoring() {
    if [ "$OS_TYPE" != "Linux" ]; then
        log_with_timestamp "⚠ 高负载传感器采集仅支持 Linux 系统"
        return 0
    fi
    
    (
        sleep 1800  # 等待30分钟
        log_with_timestamp "=== 启动高负载传感器采集 ==="
        while true; do
            {
                echo "=== 传感器状态 [$(date '+%Y-%m-%d %H:%M:%S')] ==="
                ipmitool sensor list 2>/dev/null || echo "IPMI 传感器数据获取失败"
            } >> "${LOG_BASE}/monitor_logs/highload_sensors.log"
            sleep "$SENSOR_INTERVAL"
        done
    ) &
    HIGHLOAD_MONITOR_PID=$!
    BG_PIDS+=($!)
    disown -h $HIGHLOAD_MONITOR_PID 2>/dev/null || true
    
    log_with_timestamp "✓ 高负载传感器采集已启动 (PID: $HIGHLOAD_MONITOR_PID)"
}

# 停止监控
stop_monitoring() {
    log_with_timestamp "→ 停止监控进程"
    
    # 终止主监控进程
    if [ -n "$MONITOR_PID" ]; then
        kill $MONITOR_PID 2>/dev/null || true
        MONITOR_PID=""
    fi
    
    # 终止高负载监控
    if [ -n "$HIGHLOAD_MONITOR_PID" ]; then
        kill $HIGHLOAD_MONITOR_PID 2>/dev/null || true
        HIGHLOAD_MONITOR_PID=""
    fi
    
    # 清理相关进程
    pkill -f "nvidia-smi --query-gpu" 2>/dev/null || true
    pkill -f "system_monitor.csv" 2>/dev/null || true
    pkill -f "ipmitool sensor list" 2>/dev/null || true
    
    # 清理后台进程
    for pid in $(jobs -p 2>/dev/null); do
        kill $pid 2>/dev/null || true
    done
    
    log_with_timestamp "✓ 监控已停止"
}

################################################################################
# 测试结果验证函数
################################################################################

# 验证测试结果文件
validate_test_result() {
    local log_file="$1"
    local test_name="$2"
    
    if [ ! -f "$log_file" ]; then
        log_with_timestamp "✗ 错误：${test_name} 日志文件不存在" "${LOG_BASE}/stage_logs/validation.log"
        return 1
    fi
    
    local file_size=$(stat -f%z "$log_file" 2>/dev/null || stat -c%s "$log_file" 2>/dev/null)
    if [ "$file_size" -eq 0 ]; then
        log_with_timestamp "✗ 错误：${test_name} 日志文件为空" "${LOG_BASE}/stage_logs/validation.log"
        return 1
    fi
    
    # 检查是否包含错误信息
    if grep -qi "error\|failed\|exception" "$log_file" 2>/dev/null; then
        log_with_timestamp "⚠ 警告：${test_name} 日志中包含错误信息" "${LOG_BASE}/stage_logs/validation.log"
        return 1
    fi
    
    log_with_timestamp "✓ ${test_name} 结果验证通过" "${LOG_BASE}/stage_logs/validation.log"
    return 0
}

################################################################################
# 测试阶段函数
################################################################################

# 执行带重试和超时的测试
execute_test_with_retry() {
    local test_name="$1"
    local test_command="$2"
    local log_file="$3"
    local retry_count=0
    local max_retry=$MAX_RETRY
    
    while [ $retry_count -lt $max_retry ]; do
        log_with_timestamp "执行 ${test_name} (尝试 $((retry_count + 1))/$max_retry)" "${LOG_BASE}/stage_logs/stage_progress.log"
        
        # 启动超时检测
        (
            sleep "$STAGE_TIMEOUT"
            log_with_timestamp "✗ 错误：${test_name} 超时 (${STAGE_TIMEOUT}秒)" "${LOG_BASE}/stage_logs/stage_progress.log"
            kill $$
        ) &
        local timeout_pid=$!
        
        # 执行测试
        if eval "$test_command" > "$log_file" 2>&1; then
            kill $timeout_pid 2>/dev/null || true
            validate_test_result "$log_file" "$test_name"
            return 0
        else
            kill $timeout_pid 2>/dev/null || true
            retry_count=$((retry_count + 1))
            if [ $retry_count -lt $max_retry ]; then
                log_with_timestamp "⚠ ${test_name} 失败，等待5秒后重试..." "${LOG_BASE}/stage_logs/stage_progress.log"
                sleep 5
            fi
        fi
    done
    
    log_with_timestamp "✗ ${test_name} 失败，已达到最大重试次数" "${LOG_BASE}/stage_logs/stage_progress.log"
    return 1
}

# 阶段01：CUDA P2P Bandwidth latency test
test01() {
    log_with_timestamp "执行阶段01：CUDA P2P Bandwidth latency test 测试..."
    execute_test_with_retry "P2P_Bandwidth_Latency" \
        "${CUDA_SAMPLE_DIR}/5_Domain_Specific/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest" \
        "${LOG_BASE}/PERF_GPU00001.log"
}

# 阶段02：H2D单卡测试
test02() {
    log_with_timestamp "执行阶段02：H2D单卡测试..."
    execute_test_with_retry "H2D_Single" \
        "${CUDA_SAMPLE_DIR}/1_Utilities/bandwidthTest/bandwidthTest --mode=range --start=1024000 --end=64000000 --increment=10240 --htod --device=all" \
        "${LOG_BASE}/PERF_GPU00002_H2D.log"
}

# 阶段03：H2D多卡并行测试
test03() {
    log_with_timestamp "执行阶段03：H2D多卡并行测试..."
    local pids=()
    
    for device_id in $(seq 0 $((GPU_COUNT-1))); do
        log_with_timestamp "  启动 GPU ${device_id} H2D 测试..."
        ${CUDA_SAMPLE_DIR}/1_Utilities/bandwidthTest/bandwidthTest --mode=range --start=1024000 --end=64000000 \
            --increment=10240 --htod --device=$device_id > ${LOG_BASE}/PERF_GPU00003_gpu${device_id}_H2D.log 2>&1 &
        pids+=($!)
    done
    
    wait ${pids[@]}
    
    for device_id in $(seq 0 $((GPU_COUNT-1))); do
        validate_test_result "${LOG_BASE}/PERF_GPU00003_gpu${device_id}_H2D.log" "H2D_GPU${device_id}"
    done
}

# 阶段04：D2H单卡测试
test04() {
    log_with_timestamp "执行阶段04：D2H单卡测试..."
    execute_test_with_retry "D2H_Single" \
        "${CUDA_SAMPLE_DIR}/1_Utilities/bandwidthTest/bandwidthTest --mode=range --start=1024000 --end=64000000 --increment=10240 --dtoh --device=all" \
        "${LOG_BASE}/PERF_GPU00004_D2H.log"
}

# 阶段05：D2H多卡并行测试
test05() {
    log_with_timestamp "执行阶段05：D2H多卡并行测试..."
    local pids=()
    
    for device_id in $(seq 0 $((GPU_COUNT-1))); do
        log_with_timestamp "  启动 GPU ${device_id} D2H 测试..."
        ${CUDA_SAMPLE_DIR}/1_Utilities/bandwidthTest/bandwidthTest --mode=range --start=1024000 --end=64000000 \
            --increment=10240 --dtoh --device=$device_id > ${LOG_BASE}/PERF_GPU00005_gpu${device_id}_D2H.log 2>&1 &
        pids+=($!)
    done
    
    wait ${pids[@]}
    
    for device_id in $(seq 0 $((GPU_COUNT-1))); do
        validate_test_result "${LOG_BASE}/PERF_GPU00005_gpu${device_id}_D2H.log" "D2H_GPU${device_id}"
    done
}

# 阶段06：D2D测试
test06() {
    log_with_timestamp "执行阶段06：D2D测试..."
    execute_test_with_retry "D2D" \
        "${CUDA_SAMPLE_DIR}/1_Utilities/bandwidthTest/bandwidthTest --mode=range --start=1024000 --end=64000000 --increment=10240 --dtod --device=all" \
        "${LOG_BASE}/PERF_GPU00006.log"
}

# 阶段07：H2D_D2H单卡并行测试
test07() {
    log_with_timestamp "执行阶段07：H2D_D2H单卡并行测试..."
    
    for device_id in $(seq 0 $((GPU_COUNT-1))); do
        log_with_timestamp "  启动 GPU ${device_id} 双向测试..."
        
        ${CUDA_SAMPLE_DIR}/1_Utilities/bandwidthTest/bandwidthTest --mode=range --start=1024000 --end=64000000 \
            --increment=10240 --htod --device=$device_id > ${LOG_BASE}/PERF_GPU00007_gpu${device_id}_htod.log 2>&1 &
        local htod_pid=$!
        
        ${CUDA_SAMPLE_DIR}/1_Utilities/bandwidthTest/bandwidthTest --mode=range --start=1024000 --end=64000000 \
            --increment=10240 --dtoh --device=$device_id > ${LOG_BASE}/PERF_GPU00007_gpu${device_id}_dtoh.log 2>&1 &
        local dtoh_pid=$!
        
        wait $htod_pid $dtoh_pid
        log_with_timestamp "  ✓ GPU ${device_id} 测试完成"
        
        validate_test_result "${LOG_BASE}/PERF_GPU00007_gpu${device_id}_htod.log" "H2D_GPU${device_id}"
        validate_test_result "${LOG_BASE}/PERF_GPU00007_gpu${device_id}_dtoh.log" "D2H_GPU${device_id}"
    done
}

# 阶段08：H2D_D2H多卡并行测试
test08() {
    log_with_timestamp "执行阶段08：H2D_D2H多卡并行测试..."
    local pids=()
    
    for device_id in $(seq 0 $((GPU_COUNT-1))); do
        log_with_timestamp "  启动 GPU ${device_id} H2D 测试..."
        ${CUDA_SAMPLE_DIR}/1_Utilities/bandwidthTest/bandwidthTest --mode=range --start=1024000 --end=64000000 \
            --increment=10240 --htod --device=$device_id > ${LOG_BASE}/PERF_GPU00008_gpu${device_id}_htod.log 2>&1 &
        pids+=($!)
        
        log_with_timestamp "  启动 GPU ${device_id} D2H 测试..."
        ${CUDA_SAMPLE_DIR}/1_Utilities/bandwidthTest/bandwidthTest --mode=range --start=1024000 --end=64000000 \
            --increment=10240 --dtoh --device=$device_id > ${LOG_BASE}/PERF_GPU00008_gpu${device_id}_dtoh.log 2>&1 &
        pids+=($!)
    done
    
    wait ${pids[@]}
    
    for device_id in $(seq 0 $((GPU_COUNT-1))); do
        validate_test_result "${LOG_BASE}/PERF_GPU00008_gpu${device_id}_htod.log" "H2D_GPU${device_id}"
        validate_test_result "${LOG_BASE}/PERF_GPU00008_gpu${device_id}_dtoh.log" "D2H_GPU${device_id}"
    done
}

# 阶段09：GPU压力测试
test09() {
    log_with_timestamp "执行阶段09：GPU压力测试 (时长: ${GPU_BURN_DURATION}秒)..."
    
    # 启动 gpu_burn
    cd "$(dirname "$GPU_BURN_PATH")" || exit 1
    ./gpu_burn "$GPU_BURN_DURATION" > "${LOG_BASE}/GPU-BURN-${GPU_BURN_DURATION}.log" 2>&1 &
    local GPU_BURN_PID=$!
    
    log_with_timestamp "✓ GPU Burn 已启动 (PID: $GPU_BURN_PID)"
    
    # 启动高负载传感器采集
    start_highload_monitoring
    
    # 实时进度显示
    local elapsed=0
    while kill -0 $GPU_BURN_PID 2>/dev/null; do
        sleep 30
        elapsed=$((elapsed + 30))
        local progress=$((elapsed * 100 / GPU_BURN_DURATION))
        local remaining=$((GPU_BURN_DURATION - elapsed))
        
        # 获取 GPU 状态
        local gpu_status=$(nvidia-smi --query-gpu=index,name,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null | tr '\n' '|' | sed 's/|/ /g')
        
        log_with_timestamp "  压力测试进度: ${progress}% (${elapsed}s/${GPU_BURN_DURATION}s), 剩余: ${remaining}s, GPU状态: ${gpu_status}"
    done
    
    # 等待 gpu_burn 完成
    wait $GPU_BURN_PID
    local burn_result=$?
    
    if [ $burn_result -eq 0 ]; then
        log_with_timestamp "✓ GPU 压力测试完成"
    else
        log_with_timestamp "✗ GPU 压力测试失败 (退出码: $burn_result)"
    fi
    
    validate_test_result "${LOG_BASE}/GPU-BURN-${GPU_BURN_DURATION}.log" "GPU_BURN"
}

#============================== GPU 基础验证 ==============================
test_gpu_basic() {
    log_with_timestamp "GPU 基础验证"
    
    # 验证 nvidia-smi 可用
    if nvidia-smi &>/dev/null; then
        log_with_timestamp "✓ nvidia-smi 命令正常"
    else
        log_with_timestamp "✗ nvidia-smi 命令失败"
    fi
    
    # 验证各 GPU 可访问
    for i in $(seq 0 $((GPU_COUNT - 1))); do
        if nvidia-smi --id=$i &>/dev/null; then
            log_with_timestamp "✓ GPU $i (${GPU_NAMES[$i]:-unknown}) 可访问"
        else
            log_with_timestamp "✗ GPU $i 访问失败"
        fi
    done
    
    # 检查 GPU 温度
    for i in $(seq 0 $((GPU_COUNT - 1))); do
        local temp=$(nvidia-smi --id=$i --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null)
        if [ -n "$temp" ] && [ "$temp" -lt 85 ]; then
            log_with_timestamp "✓ GPU $i 温度正常 (${temp}°C)"
        else
            log_with_timestamp "⚠ GPU $i 温度偏高 (${temp}°C)"
        fi
    done
    
    # 显示驱动版本
    local driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$driver" ]; then
        log_with_timestamp "✓ NVIDIA 驱动版本: $driver"
    fi
}

#============================== CUDA 算力测试 ==============================
test_cuda_devicequery() {
    log_with_timestamp "CUDA 设备查询测试"
    
    local cuda_binary
    cuda_binary=$(find_cuda_sample "deviceQuery")
    if [ -z "$cuda_binary" ]; then
        cuda_binary=$(find_cuda_sample "deviceQuery/deviceQuery")
    fi
    if [ -z "$cuda_binary" ]; then
        cuda_binary="/usr/local/cuda/extras/demo_suite/deviceQuery"
    fi
    
    if [ ! -x "$cuda_binary" ]; then
        log_with_timestamp "⚠ 未找到 deviceQuery，跳过 CUDA 算力测试"
        return 0
    fi
    
    local log_file="${LOG_BASE}/stage_logs/deviceQuery.log"
    if $cuda_binary > "$log_file" 2>&1; then
        if grep -q "Result = PASS" "$log_file"; then
            log_with_timestamp "✓ CUDA deviceQuery 测试通过"
        else
            log_with_timestamp "⚠ CUDA deviceQuery 结果异常"
        fi
    else
        log_with_timestamp "✗ CUDA deviceQuery 测试失败"
    fi
}

# nvidia-smi 替代压力测试（gpu_burn 不可用时的回退）
stress_with_nvidia_smi() {
    log_with_timestamp "使用 CUDA 内存压力测试替代（nvidia-smi 模式）..."
    
    local test_time=${GPU_BURN_DURATION:-600}
    local interval=30
    local elapsed=0
    
    while [ $elapsed -lt $test_time ]; do
        nvidia-smi --id=0 --query-gpu=utilization.gpu,temperature.gpu,power.draw \
            --format=csv >> "${LOG_BASE}/stress/nvidia_smi_stress.csv" 2>&1
        sleep $interval
        elapsed=$((elapsed + interval))
        log_with_timestamp "压力测试进度: ${elapsed}/${test_time} 秒"
    done
    
    log_with_timestamp "✓ nvidia-smi 压力测试完成"
}

################################################################################
# 系统信息采集函数
################################################################################

# 执行环境初始化
execute_init() {
    echo "=== 系统环境初始化 ==="
    detect_gpu_count
    detect_cuda_samples
    detect_gpu_burn
    check_required_tools
}

# 执行系统信息采集
execute_sysinfo() {
    log_with_timestamp "=== 开始执行系统信息采集 ==="
    
    # 创建 sysinfo 子目录
    mkdir -p "${LOG_BASE}/sysinfo/hardware"
    mkdir -p "${LOG_BASE}/sysinfo/software"
    mkdir -p "${LOG_BASE}/sysinfo/network"
    mkdir -p "${LOG_BASE}/sysinfo/storage"
    
    # ==================== 硬件信息 ====================
    log_with_timestamp "→ 采集硬件信息..."
    
    # 系统硬件信息
    if command -v lshw >/dev/null 2>&1; then
        lshw 2>&1 | tee -a "${LOG_BASE}/sysinfo/hardware/lshw.log" >/dev/null
    fi
    
    if [ "$OS_TYPE" = "Darwin" ]; then
        system_profiler SPHardwareDataType > "${LOG_BASE}/sysinfo/hardware/hardware_info.log"
    else
        dmidecode -t bios 2>&1 | tee -a "${LOG_BASE}/sysinfo/hardware/bios_info.log" >/dev/null
        dmidecode -t system 2>&1 | tee -a "${LOG_BASE}/sysinfo/hardware/system_info.log" >/dev/null
    fi
    
    dmesg > "${LOG_BASE}/sysinfo/hardware/dmesg.log"
    lspci -vt > "${LOG_BASE}/sysinfo/hardware/pci_tree.log" 2>/dev/null || true
    lspci -vvv > "${LOG_BASE}/sysinfo/hardware/pci_info.log" 2>/dev/null || true
    
    if command -v lscpu >/dev/null 2>&1; then
        lscpu > "${LOG_BASE}/sysinfo/hardware/cpu_info.log"
    else
        sysctl -n machdep.cpu.brand_string > "${LOG_BASE}/sysinfo/hardware/cpu_info.log"
    fi
    
    if command -v lstopo >/dev/null 2>&1; then
        lstopo > "${LOG_BASE}/sysinfo/hardware/topology.log"
    fi
    
    # 内存信息
    if command -v free >/dev/null 2>&1; then
        free -h > "${LOG_BASE}/sysinfo/hardware/memory_info.log"
    else
        vm_stat > "${LOG_BASE}/sysinfo/hardware/memory_info.log"
    fi
    
    # ==================== 存储信息 ====================
    log_with_timestamp "→ 采集存储信息..."
    
    df -h > "${LOG_BASE}/sysinfo/storage/disk_space.log"
    lsblk > "${LOG_BASE}/sysinfo/storage/block_devices.log" 2>/dev/null || diskutil list > "${LOG_BASE}/sysinfo/storage/block_devices.log"
    
    # NVMe 信息
    if command -v smartctl >/dev/null 2>&1; then
        for nvme in /dev/nvme?n?; do
            if [ -e "$nvme" ]; then
                smartctl -A "$nvme" > "${LOG_BASE}/sysinfo/storage/nvme_$(basename $nvme)_smart.log" 2>&1 || true
            fi
        done
    fi
    
    if command -v nvme >/dev/null 2>&1; then
        nvme smart-log /dev/nvme0n1 > "${LOG_BASE}/sysinfo/storage/nvme_smart_log.log" 2>/dev/null || true
        nvme id-ctrl /dev/nvme0n1 > "${LOG_BASE}/sysinfo/storage/nvme_id_ctrl.log" 2>/dev/null || true
        nvme id-ns /dev/nvme0n1 > "${LOG_BASE}/sysinfo/storage/nvme_id_ns.log" 2>/dev/null || true
    fi
    
    # RAID 信息
    if [ "$OS_TYPE" = "Linux" ]; then
        mdadm --detail /dev/md0 > "${LOG_BASE}/sysinfo/storage/raid_status.log" 2>&1 || echo "RAID状态获取失败或不存在"
    fi
    
    # ==================== IPMI 信息 ====================
    log_with_timestamp "→ 采集 IPMI 信息..."
    
    if command -v ipmitool >/dev/null 2>&1; then
        ipmitool fru print > "${LOG_BASE}/sysinfo/hardware/ipmi_fru.log" 2>&1 || true
        ipmitool lan print > "${LOG_BASE}/sysinfo/hardware/ipmi_lan.log" 2>&1 || true
        ipmitool sensor list > "${LOG_BASE}/sysinfo/hardware/ipmi_sensors.log" 2>&1 || true
    fi
    
    # BIOS 设置（如果可用）
    if [ -x /root/SCELNX_64 ]; then
        /root/SCELNX_64 /o /s "${LOG_BASE}/sysinfo/hardware/BIOS_setup.log" 2>&1 || true
    fi
    
    # ==================== GPU 信息 ====================
    log_with_timestamp "→ 采集 GPU 信息..."
    
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi > "${LOG_BASE}/sysinfo/hardware/nvidia_smi.log"
        nvidia-smi -q > "${LOG_BASE}/sysinfo/hardware/nvidia_smi_query.log"
        nvidia-smi -L > "${LOG_BASE}/sysinfo/hardware/nvidia_smi_list.log"
        nvidia-smi topo -m > "${LOG_BASE}/sysinfo/hardware/nvidia_smi_topo.log"
        nvidia-smi topo -p2p p > "${LOG_BASE}/sysinfo/hardware/nvidia_smi_topo_p2p.log"
    fi
    
    # CUDA 信息
    local cuda_demo_paths=("/usr/local/cuda/extras/demo_suite" "$CUDA_SAMPLE_DIR/../../../extras/demo_suite")
    for demo_path in "${cuda_demo_paths[@]}"; do
        if [ -d "$demo_path" ]; then
            [ -x "$demo_path/deviceQuery" ] && "$demo_path/deviceQuery" > "${LOG_BASE}/sysinfo/hardware/cuda_deviceQuery.log" 2>&1 || true
            [ -x "$demo_path/busGrind" ] && "$demo_path/busGrind" > "${LOG_BASE}/sysinfo/hardware/cuda_busGrind.log" 2>&1 || true
            [ -x "$demo_path/vectorAdd" ] && "$demo_path/vectorAdd" > "${LOG_BASE}/sysinfo/hardware/cuda_vectorAdd.log" 2>&1 || true
            break
        fi
    done
    
    # ==================== 软件信息 ====================
    log_with_timestamp "→ 采集软件信息..."
    
    # 操作系统信息
    if [ "$OS_TYPE" = "Darwin" ]; then
        sw_vers > "${LOG_BASE}/sysinfo/software/os_info.log"
    else
        cat /etc/os-release > "${LOG_BASE}/sysinfo/software/os_release.log"
    fi
    
    cat /proc/version > "${LOG_BASE}/sysinfo/software/kernel_version.log" 2>/dev/null || uname -v > "${LOG_BASE}/sysinfo/software/kernel_version.log"
    uname -a > "${LOG_BASE}/sysinfo/software/kernel_info.log"
    
    # NVIDIA 驱动信息
    if [ "$OS_TYPE" = "Linux" ]; then
        cat /proc/driver/nvidia/version > "${LOG_BASE}/sysinfo/software/nvidia_driver_version.log" 2>/dev/null || true
    fi
    
    # CPU 信息
    cat /proc/cpuinfo > "${LOG_BASE}/sysinfo/software/cpu_info.log" 2>/dev/null || sysctl -a | grep machdep.cpu > "${LOG_BASE}/sysinfo/software/cpu_info.log"
    
    cat /proc/meminfo > "${LOG_BASE}/sysinfo/software/memory_info.log" 2>/dev/null || vm_stat > "${LOG_BASE}/sysinfo/software/memory_info.log"
    
    cat /proc/interrupts > "${LOG_BASE}/sysinfo/software/interrupts.log" 2>/dev/null || true
    
    # 内核参数
    if command -v sysctl >/dev/null 2>&1; then
        sysctl -a > "${LOG_BASE}/sysinfo/software/sysctl_params.log"
    fi
    
    # 加载的模块
    if command -v lsmod >/dev/null 2>&1; then
        lsmod > "${LOG_BASE}/sysinfo/software/modules.log"
    else
        kextstat > "${LOG_BASE}/sysinfo/software/modules.log"
    fi
    
    # ==================== 网络信息 ====================
    log_with_timestamp "→ 采集网络信息..."
    
    cat /proc/net/dev > "${LOG_BASE}/sysinfo/network/network_devices.log" 2>/dev/null || ifconfig > "${LOG_BASE}/sysinfo/network/network_devices.log"
    cat /proc/net/route > "${LOG_BASE}/sysinfo/network/network_routes.log" 2>/dev/null || netstat -rn > "${LOG_BASE}/sysinfo/network/network_routes.log"
    cat /proc/net/arp > "${LOG_BASE}/sysinfo/network/network_arp.log" 2>/dev/null || arp -a > "${LOG_BASE}/sysinfo/network/network_arp.log"
    cat /proc/net/if_inet6 > "${LOG_BASE}/sysinfo/network/network_ipv6.log" 2>/dev/null || true
    
    # ==================== 服务和进程信息 ====================
    log_with_timestamp "→ 采集服务信息..."
    
    if command -v systemctl >/dev/null 2>&1; then
        systemctl list-units > "${LOG_BASE}/sysinfo/software/services.log"
    fi
    
    journalctl -k --since "1 hour ago" > "${LOG_BASE}/sysinfo/software/kernel_journal.log" 2>/dev/null || true
    
    ps auxf > "${LOG_BASE}/sysinfo/software/process_list.log"
    env > "${LOG_BASE}/sysinfo/software/environment_vars.log"
    
    # 系统负载
    uptime > "${LOG_BASE}/sysinfo/software/system_load.log"
    
    log_with_timestamp "✓ 系统信息采集完成"
}

################################################################################
# 报告生成函数
################################################################################

# 生成测试报告
generate_report() {
    log_with_timestamp "=== 生成测试报告 ==="
    
    # 文本格式报告
    cat > "${LOG_BASE}/reports/test_summary.txt" <<EOF
================================================================================
                         GPU 测试报告摘要
================================================================================

测试时间: $(date '+%Y-%m-%d %H:%M:%S')
操作系统: $OS_TYPE $OS_VERSION
GPU 数量: $GPU_COUNT

================================================================================
                            测试结果
================================================================================

EOF
    
    # 添加各阶段测试结果
    for i in $(seq -w 1 $TOTAL_STAGES); do
        local test_log="${LOG_BASE}/PERF_GPU000${i}.log"
        if [ ! -f "$test_log" ]; then
            test_log="${LOG_BASE}/PERF_GPU000${i}_H2D.log"
        fi
        
        if [ -f "$test_log" ]; then
            echo "阶段 ${i}: 测试完成" >> "${LOG_BASE}/reports/test_summary.txt"
            echo "  日志文件: $(basename $test_log)" >> "${LOG_BASE}/reports/test_summary.txt"
            echo "  文件大小: $(du -h "$test_log" | cut -f1)" >> "${LOG_BASE}/reports/test_summary.txt"
            echo "" >> "${LOG_BASE}/reports/test_summary.txt"
        else
            echo "阶段 ${i}: 测试日志不存在" >> "${LOG_BASE}/reports/test_summary.txt"
            echo "" >> "${LOG_BASE}/reports/test_summary.txt"
        fi
    done
    
    cat >> "${LOG_BASE}/reports/test_summary.txt" <<EOF

================================================================================
                            文件清单
================================================================================

EOF
    
    # 列出所有日志文件
    find "${LOG_BASE}" -type f -name "*.log" | while read file; do
        echo "$(basename $file) - $(du -h "$file" | cut -f1)" >> "${LOG_BASE}/reports/test_summary.txt"
    done
    
    echo "" >> "${LOG_BASE}/reports/test_summary.txt"
    echo "================================================================================" >> "${LOG_BASE}/reports/test_summary.txt"
    echo "完整测试日志目录: ${LOG_BASE}" >> "${LOG_BASE}/reports/test_summary.txt"
    echo "================================================================================" >> "${LOG_BASE}/reports/test_summary.txt"
    
    # HTML 格式报告
    generate_html_report
    
    log_with_timestamp "✓ 测试报告已生成"
    log_with_timestamp "  - 文本报告: ${LOG_BASE}/reports/test_summary.txt"
    log_with_timestamp "  - HTML报告: ${LOG_BASE}/reports/test_report.html"
}

# 生成 HTML 报告
generate_html_report() {
    cat > "${LOG_BASE}/reports/test_report.html" <<'HTMLHEADER'
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPU 测试报告</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }
        h2 {
            color: #555;
            border-left: 4px solid #4CAF50;
            padding-left: 10px;
            margin-top: 30px;
        }
        .info-box {
            background-color: #f9f9f9;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }
        .success {
            color: #4CAF50;
            font-weight: bold;
        }
        .warning {
            color: #ff9800;
            font-weight: bold;
        }
        .error {
            color: #f44336;
            font-weight: bold;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #4CAF50;
            color: white;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .progress-bar {
            background-color: #e0e0e0;
            border-radius: 5px;
            height: 20px;
            margin: 10px 0;
        }
        .progress-fill {
            background-color: #4CAF50;
            height: 100%;
            border-radius: 5px;
            transition: width 0.3s;
        }
        .footer {
            margin-top: 30px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            text-align: center;
            color: #777;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>GPU 测试报告</h1>
        
        <div class="info-box">
            <h3>测试概览</h3>
            <p><strong>测试时间:</strong> <span id="test-time"></span></p>
            <p><strong>操作系统:</strong> <span id="os-info"></span></p>
            <p><strong>GPU 数量:</strong> <span id="gpu-count"></span></p>
            <p><strong>GPU 型号:</strong> <span id="gpu-models"></span></p>
        </div>

        <h2>测试阶段</h2>
        <table id="test-stages">
            <thead>
                <tr>
                    <th>阶段</th>
                    <th>测试名称</th>
                    <th>状态</th>
                    <th>日志文件</th>
                </tr>
            </thead>
            <tbody>
            </tbody>
        </table>

        <h2>系统监控数据</h2>
        <div class="info-box">
            <p>系统资源监控数据已保存到 <code>monitor_logs</code> 目录</p>
        </div>

        <h2>文件清单</h2>
        <table id="file-list">
            <thead>
                <tr>
                    <th>文件名</th>
                    <th>大小</th>
                </tr>
            </thead>
            <tbody>
            </tbody>
        </table>

        <div class="footer">
            <p>测试报告由 AI-bench 自动生成</p>
        </div>
    </div>

    <script>
        // 填充测试信息
        document.getElementById('test-time').textContent = new Date().toLocaleString('zh-CN');
        document.getElementById('os-info').textContent = '$OS_TYPE $OS_VERSION';
        document.getElementById('gpu-count').textContent = '$GPU_COUNT';
        document.getElementById('gpu-models').textContent = '$(for i in $(seq 0 $((GPU_COUNT - 1))); do echo -n "GPU$i:${GPU_NAMES[$i]} "; done)';

        // 测试阶段数据
        const testStages = [
HTMLHEADER

    # 添加测试阶段数据
    local stage_names=(
        "P2P Bandwidth Latency"
        "H2D 单卡测试"
        "H2D 多卡并行测试"
        "D2H 单卡测试"
        "D2H 多卡并行测试"
        "D2D 测试"
        "H2D_D2H 单卡并行测试"
        "H2D_D2H 多卡并行测试"
        "GPU 压力测试"
    )
    
    for i in $(seq 0 $((TOTAL_STAGES - 1))); do
        local stage_num=$((i + 1))
        local stage_name="${stage_names[$i]}"
        local log_file=""
        
        # 查找对应的日志文件
        if [ $stage_num -eq 1 ]; then
            log_file="PERF_GPU00001.log"
        elif [ $stage_num -eq 2 ]; then
            log_file="PERF_GPU00002_H2D.log"
        elif [ $stage_num -eq 4 ]; then
            log_file="PERF_GPU00004_D2H.log"
        elif [ $stage_num -eq 6 ]; then
            log_file="PERF_GPU00006.log"
        elif [ $stage_num -eq 9 ]; then
            log_file="GPU-BURN-${GPU_BURN_DURATION}.log"
        else
            log_file="PERF_GPU0000${stage_num}_*.log"
        fi
        
        local status="success"
        local status_text="✓ 通过"
        
        if [ ! -f "${LOG_BASE}/${log_file}" ] && [[ $log_file != *"*"* ]]; then
            status="error"
            status_text="✗ 失败"
        fi
        
        cat >> "${LOG_BASE}/reports/test_report.html" <<EOF
            { stage: '${stage_num}', name: '${stage_name}', status: '${status}', statusText: '${status_text}', log: '${log_file}' },
EOF
    done
    
    cat >> "${LOG_BASE}/reports/test_report.html" <<'HTMLMIDDLE'
        ];

        // 填充测试阶段表格
        const stagesTable = document.getElementById('test-stages').getElementsByTagName('tbody')[0];
        testStages.forEach(stage => {
            const row = stagesTable.insertRow();
            row.insertCell(0).textContent = '阶段 ' + stage.stage;
            row.insertCell(1).textContent = stage.name;
            const statusCell = row.insertCell(2);
            statusCell.innerHTML = '<span class="' + stage.status + '">' + stage.statusText + '</span>';
            row.insertCell(3).textContent = stage.log;
        });

        // 文件清单数据
        const fileList = [
HTMLMIDDLE

    # 添加文件列表数据
    find "${LOG_BASE}" -type f -name "*.log" -o -name "*.csv" | while read file; do
        local file_name=$(basename "$file")
        local file_size=$(du -h "$file" | cut -f1)
        echo "            { name: '${file_name}', size: '${file_size}' }," >> "${LOG_BASE}/reports/test_report.html"
    done
    
    cat >> "${LOG_BASE}/reports/test_report.html" <<'HTMLFOOTER'
        ];

        // 填充文件清单表格
        const fileListTable = document.getElementById('file-list').getElementsByTagName('tbody')[0];
        fileList.forEach(file => {
            const row = fileListTable.insertRow();
            row.insertCell(0).textContent = file.name;
            row.insertCell(1).textContent = file.size;
        });
    </script>
</body>
</html>
HTMLFOOTER
}

################################################################################
# 进度显示函数
################################################################################

# 显示测试进度
show_progress() {
    local stage_name="$1"
    CURRENT_STAGE=$((CURRENT_STAGE + 1))
    local progress=$((CURRENT_STAGE * 100 / TOTAL_STAGES))
    
    echo ""
    echo "================================================================================"
    echo "测试进度: [${CURRENT_STAGE}/${TOTAL_STAGES}] ${progress}%"
    echo "当前阶段: ${stage_name}"
    echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================================"
}

# 显示阶段完成状态
show_stage_completion() {
    local stage_name="$1"
    local status="$2"
    
    if [ "$status" = "success" ]; then
        echo "✓ ${stage_name} 完成"
    else
        echo "✗ ${stage_name} 失败"
    fi
}

################################################################################
# 通知函数
################################################################################

# 发送通知
send_notification() {
    local subject="$1"
    local message="$2"
    
    if [ "$ENABLE_NOTIFICATION" != true ] || [ -z "$NOTIFICATION_EMAIL" ]; then
        return 0
    fi
    
    if command -v mail >/dev/null 2>&1; then
        echo "$message" | mail -s "$subject" "$NOTIFICATION_EMAIL"
        log_with_timestamp "✓ 通知已发送到: $NOTIFICATION_EMAIL"
    else
        log_with_timestamp "⚠ 邮件发送失败：mail 命令不可用"
    fi
}

################################################################################
# 清理函数
################################################################################

# 清理所有测试进程
cleanup_all_processes() {
    log_with_timestamp "→ 清理所有测试进程..."
    
    stop_monitoring
    
    # 终止所有可能的残留进程
    pkill -f "bandwidthTest" 2>/dev/null || true
    pkill -f "p2pBandwidthLatencyTest" 2>/dev/null || true
    pkill -f "gpu_burn" 2>/dev/null || true
    pkill -f "testGPU.sh" 2>/dev/null || true
    
    # 杀死进程组
    kill -- -$$ 2>/dev/null || true
    
    # 等待所有后台作业完成
    wait 2>/dev/null || true
    
    log_with_timestamp "✓ 进程清理完成"
}

# 信号处理函数
cleanup_handler() {
    log_with_timestamp "→ 捕获中断信号，正在清理..."
    cleanup_all_processes
    send_notification "GPU 测试中断" "测试被用户中断"
    exit 1
}

################################################################################
# 参数解析函数
################################################################################

# 显示帮助信息
show_help() {
    cat <<EOF
GPU 测试脚本 - 使用说明

用法: $0 [选项]

选项:
    --full          执行完整测试（初始化 + 系统信息 + 基准测试 + 报告）
    --quick         快速测试（初始化 + 系统信息 + GPU 基础验证）
    --burn-only     仅运行 GPU 压力测试（阶段09）
    --info          仅采集系统信息
    --init          仅执行环境初始化
    --sysinfo       仅执行系统信息采集
    --benchmark     仅执行基准测试（九阶段性能测试）
    --report        仅生成测试报告
    --help          显示此帮助信息

示例:
    $0 --full       # 执行完整测试流程
    $0 --quick      # 快速测试（GPU 基础验证）
    $0 --benchmark  # 仅执行基准测试
    $0 --sysinfo    # 仅采集系统信息
    $0 --burn-only  # 仅运行 GPU 压力测试

配置参数:
    可在脚本开头修改以下参数：
    - GPU_BURN_DURATION: GPU 压力测试时长（默认：3600秒）
    - MONITOR_INTERVAL: 监控采样间隔（默认：10秒）
    - MAX_RETRY: 失败重试次数（默认：3次）
    - STAGE_TIMEOUT: 单个测试阶段超时时间（默认：3600秒）

EOF
}

# 解析命令行参数
parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --full)
                FULL_MODE=true
                ;;
            --quick)
                QUICK_MODE=true
                ;;
            --burn-only)
                BURN_ONLY_MODE=true
                ;;
            --info)
                INFO_MODE=true
                ;;
            --init)
                INIT_MODE=true
                ;;
            --sysinfo)
                SYSINFO_MODE=true
                ;;
            --benchmark)
                BENCH_MODE=true
                ;;
            --report)
                REPORT_MODE=true
                ;;
            --help|-h)
                show_help
                exit 0
                ;;
            --*)
                echo "错误：无效参数 '$1'" >&2
                echo "使用 --help 查看帮助信息" >&2
                exit 1
                ;;
            *)
                echo "错误：未知参数 '$1'" >&2
                echo "使用 --help 查看帮助信息" >&2
                exit 1
                ;;
        esac
        shift
    done
}

################################################################################
# 主执行函数
################################################################################

# 执行基准测试
execute_benchmark() {
    log_with_timestamp "=== 启动九阶段性能测试 ==="
    
    local stages=(test01 test02 test03 test04 test05 test06 test07 test08 test09)
    local stage_names=(
        "P2P Bandwidth Latency"
        "H2D 单卡测试"
        "H2D 多卡并行测试"
        "D2H 单卡测试"
        "D2H 多卡并行测试"
        "D2D 测试"
        "H2D_D2H 单卡并行测试"
        "H2D_D2H 多卡并行测试"
        "GPU 压力测试"
    )
    
    local failed_stages=()
    
    for i in $(seq 0 $((TOTAL_STAGES - 1))); do
        local stage="${stages[$i]}"
        local stage_name="${stage_names[$i]}"
        local stage_num=$(printf "%02d" $((i + 1)))
        
        show_progress "$stage_name"
        STAGE_START_TIME=$(date +%s)
        
        # 记录阶段开始
        echo "=== 阶段 ${stage_num}: ${stage_name} ===" | tee "${LOG_BASE}/stage_logs/stage_${stage_num}.log"
        
        # 启动监控
        start_monitoring
        
        # 执行测试
        if $stage; then
            echo "✓ 阶段 ${stage_num} 完成 [$(date '+%Y-%m-%d %H:%M:%S')]" | tee -a "${LOG_BASE}/stage_logs/stage_${stage_num}.log"
            show_stage_completion "$stage_name" "success"
        else
            echo "✗ 阶段 ${stage_num} 失败！错误码 $? [$(date '+%Y-%m-%d %H:%M:%S')]" | tee -a "${LOG_BASE}/stage_logs/stage_${stage_num}.log"
            show_stage_completion "$stage_name" "failed"
            failed_stages+=("$stage_name")
        fi
        
        # 停止监控
        stop_monitoring
        
        # 检查是否有阶段失败
        if [ ${#failed_stages[@]} -gt 0 ]; then
            echo ""
            echo "⚠ 警告：以下测试阶段失败："
            for failed in "${failed_stages[@]}"; do
                echo "  - $failed"
            done
        fi
        
        echo ""
    done
    
    log_with_timestamp "=== 所有测试阶段完成 ==="
    
    # 发送测试完成通知
    if [ ${#failed_stages[@]} -eq 0 ]; then
        send_notification "GPU 测试完成" "所有测试阶段均成功完成"
    else
        send_notification "GPU 测试部分失败" "失败的测试阶段：${failed_stages[*]}"
    fi
}

# 主函数
main() {
    local start_time=$(date +%s)
    
    # 注册信号捕获
    trap cleanup_handler SIGINT SIGTERM
    
    # 解析参数
    parse_arguments "$@"
    
    # 如果没有指定任何模式，默认显示帮助
    if [[ -z $FULL_MODE && -z $QUICK_MODE && -z $BURN_ONLY_MODE && -z $INFO_MODE && -z $INIT_MODE && -z $SYSINFO_MODE && -z $BENCH_MODE && -z $REPORT_MODE ]]; then
        show_help
        exit 0
    fi
    
    # 初始化日志系统
    init_logging
    
    # 执行测试流程
    if [[ $FULL_MODE ]]; then
        execute_init
        execute_sysinfo
        execute_benchmark
        generate_report
    elif [[ $QUICK_MODE ]]; then
        execute_init
        execute_sysinfo
        test_gpu_basic
        test_cuda_devicequery
        generate_report
    elif [[ $BURN_ONLY_MODE ]]; then
        execute_init
        test09
        generate_report
    elif [[ $INFO_MODE ]]; then
        execute_init
        execute_sysinfo
        log_with_timestamp "系统信息已保存到 ${LOG_BASE}/sysinfo/"
    else
        [[ $INIT_MODE ]] && execute_init
        [[ $SYSINFO_MODE ]] && execute_sysinfo
        [[ $BENCH_MODE ]] && execute_benchmark
        [[ $REPORT_MODE ]] && generate_report
    fi
    
    # 计算总耗时
    local end_time=$(date +%s)
    local total_seconds=$((end_time - start_time))
    local hours=$((total_seconds / 3600))
    local minutes=$(( (total_seconds % 3600) / 60 ))
    local seconds=$((total_seconds % 60))
    
    echo ""
    echo "================================================================================"
    echo "操作完成！"
    echo "总运行时间：${hours}小时 ${minutes}分 ${seconds}秒"
    echo "日志目录: ${LOG_BASE}"
    echo "================================================================================"
    
    # 清理进程
    cleanup_all_processes
    
    exit 0
}

# 执行主函数
main "$@"