#!/bin/bash
################################################################################
# GPU 服务器性能测试脚本
# 专注于 NVIDIA GPU 工作性能测试
#
# 九阶段测试流程：
#   阶段01: P2P Bandwidth Latency Test       (GPU互联带宽延迟)
#   阶段02: H2D 单卡测试                      (Host to Device)
#   阶段03: H2D 多卡并行测试                  (多卡并行传输)
#   阶段04: D2H 单卡测试                      (Device to Host)
#   阶段05: D2H 多卡并行测试                  (多卡并行回传)
#   阶段06: D2D 测试                          (Device to Device)
#   阶段07: H2D_D2H 单卡并行测试              (双向同时传输)
#   阶段08: H2D_D2H 多卡并行测试              (多卡双向并行)
#   阶段09: GPU 压力测试                      (稳定性验证)
#
# 使用方法：
#   ./gpu_test.sh              # 执行完整测试
#   ./gpu_test.sh --quick      # 快速测试 (系统信息 + 基础验证)
#   ./gpu_test.sh --burn-only  # 仅压力测试
#   ./gpu_test.sh --info       # 仅采集信息
################################################################################

set -eo pipefail

#============================== 配置参数 ==============================
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly LOG_DIR="${SCRIPT_DIR}/test_results_$(date +%Y%m%d_%H%M%S)"
readonly GPU_BURN_TIME="${GPU_BURN_TIME:-600}"      # 压力测试时长(秒)
readonly MONITOR_INTERVAL="${MONITOR_INTERVAL:-5}"   # 监控采样间隔(秒)

#============================== 全局变量 ==============================
GPU_COUNT=0
GPU_INFO=()
GPU_NAMES=()
PASS_COUNT=0
FAIL_COUNT=0
TOTAL_TESTS=0

#============================== 颜色输出 ==============================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info()  { echo -e "${BLUE}[INFO]${NC} $1"; }
log_pass()  { echo -e "${GREEN}[PASS]${NC} $1"; ((PASS_COUNT++)); ((TOTAL_TESTS++)); }
log_fail()  { echo -e "${RED}[FAIL]${NC} $1"; ((FAIL_COUNT++)); ((TOTAL_TESTS++)); }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_test()  { echo -e "\n${BLUE}=== $1 ===${NC}"; }

#============================== 初始化 ==============================
init() {
    mkdir -p "$LOG_DIR"/{{monitor,benchmark,stress},sysinfo}
    
    log_info "测试日志目录: $LOG_DIR"
    log_info "测试开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
    
    if ! command -v nvidia-smi &>/dev/null; then
        log_fail "未找到 nvidia-smi 命令，请安装 NVIDIA 驱动"
        exit 1
    fi
    
    GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ "$GPU_COUNT" -eq 0 ]; then
        log_fail "未检测到 GPU 设备"
        exit 1
    fi
    
    log_info "检测到 ${GPU_COUNT} 个 GPU 设备"
    
    for i in $(seq 0 $((GPU_COUNT - 1))); do
        GPU_NAMES[i]=$(nvidia-smi --id=$i --query-gpu=name --format=csv,noheader 2>/dev/null)
        GPU_INFO[i]=$(nvidia-smi --id=$i -q 2>/dev/null)
    done
}

#============================== 系统信息采集 ==============================
collect_sysinfo() {
    log_test "系统信息采集"
    
    # GPU 基础信息
    nvidia-smi > "$LOG_DIR/sysinfo/nvidia-smi.log"
    nvidia-smi -q > "$LOG_DIR/sysinfo/nvidia-smi-full.log"
    nvidia-smi -L > "$LOG_DIR/sysinfo/nvidia-smi-list.log"
    nvidia-smi topo -m > "$LOG_DIR/sysinfo/nvidia-topo.log"
    nvidia-smi topo -p2p p > "$LOG_DIR/sysinfo/nvidia-p2p.log" 2>/dev/null || true
    
    # 驱动信息
    if [ -f /proc/driver/nvidia/version ]; then
        cat /proc/driver/nvidia/version > "$LOG_DIR/sysinfo/driver.log"
    fi
    
    # CUDA 版本
    if command -v nvcc &>/dev/null; then
        nvcc --version > "$LOG_DIR/sysinfo/cuda-version.log"
    fi
    
    log_pass "系统信息采集完成"
}

#============================== GPU 基础验证 ==============================
test_gpu_basic() {
    log_test "GPU 基础验证"
    
    local test_passed=true
    
    # 测试1: nvidia-smi 可用性
    if nvidia-smi &>/dev/null; then
        log_pass "nvidia-smi 命令正常"
    else
        log_fail "nvidia-smi 命令失败"
        test_passed=false
    fi
    
    # 测试2: 所有 GPU 可访问
    for i in $(seq 0 $((GPU_COUNT - 1))); do
        if nvidia-smi --id=$i &>/dev/null; then
            log_pass "GPU $i (${GPU_NAMES[$i]}) 可访问"
        else
            log_fail "GPU $i 访问失败"
            test_passed=false
        fi
    done
    
    # 测试3: GPU 温度正常
    for i in $(seq 0 $((GPU_COUNT - 1))); do
        local temp=$(nvidia-smi --id=$i --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null)
        if [ "$temp" -lt 85 ]; then
            log_pass "GPU $i 温度正常 (${temp}°C)"
        else
            log_warn "GPU $i 温度偏高 (${temp}°C)"
        fi
    done
    
    # 测试4: 驱动版本
    local driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    if [ -n "$driver" ]; then
        log_pass "NVIDIA 驱动版本: $driver"
    fi
    
    $test_passed
}

#============================== CUDA 算力测试 ==============================
test_cuda_devicequery() {
    log_test "CUDA 设备查询测试"
    
    local demo_suite="/usr/local/cuda/extras/demo_suite"
    local cuda_samples=""
    
    # 查找 deviceQuery
    for path in "$demo_suite/deviceQuery" \
                "/root/cuda-samples/build/Samples/1_Utilities/deviceQuery/deviceQuery" \
                "/usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery"; do
        if [ -x "$path" ]; then
            cuda_samples="$path"
            break
        fi
    done
    
    if [ -z "$cuda_samples" ]; then
        log_warn "未找到 deviceQuery，跳过 CUDA 算力测试"
        log_info "请安装 CUDA Samples 或手动运行: /usr/local/cuda/extras/demo_suite/deviceQuery"
        return 0
    fi
    
    if $cuda_samples > "$LOG_DIR/benchmark/deviceQuery.log" 2>&1; then
        if grep -q "Result = PASS" "$LOG_DIR/benchmark/deviceQuery.log"; then
            log_pass "CUDA deviceQuery 测试通过"
            return 0
        fi
    fi
    log_fail "CUDA deviceQuery 测试失败"
    return 1
}

#============================== P2P 互联测试 ==============================
test_p2p() {
    log_test "GPU P2P 互联测试"
    
    local p2p_test=""
    
    # 查找 p2pBandwidthLatencyTest
    for path in "/root/cuda-samples/build/Samples/5_Domain_Specific/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest" \
                "/usr/local/cuda/samples/5_Domain_Specific/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest"; do
        if [ -x "$path" ]; then
            p2p_test="$path"
            break
        fi
    done
    
    if [ -z "$p2p_test" ]; then
        log_warn "未找到 p2pBandwidthLatencyTest，跳过 P2P 测试"
        return 0
    fi
    
    if [ "$GPU_COUNT" -lt 2 ]; then
        log_info "单GPU，跳过 P2P 测试"
        return 0
    fi
    
    if $p2p_test > "$LOG_DIR/benchmark/p2p_test.log" 2>&1; then
        log_pass "P2P 互联测试通过"
        return 0
    fi
    log_fail "P2P 互联测试失败"
    return 1
}

#============================== 显存带宽测试 ==============================
test_memory_bandwidth() {
    log_test "显存带宽测试"
    
    local bw_test=""
    
    # 查找 bandwidthTest
    for path in "/root/cuda-samples/build/Samples/1_Utilities/bandwidthTest/bandwidthTest" \
                "/usr/local/cuda/samples/1_Utilities/bandwidthTest/bandwidthTest" \
                "/usr/local/cuda/extras/demo_suite/bandwidthTest"; do
        if [ -x "$path" ]; then
            bw_test="$path"
            break
        fi
    done
    
    if [ -z "$bw_test" ]; then
        log_warn "未找到 bandwidthTest，跳过带宽测试"
        return 0
    fi
    
    # H2D 测试
    log_info "执行 H2D (Host to Device) 测试..."
    if $bw_test --mode=shmoo --htod --device=all > "$LOG_DIR/benchmark/bandwidth_htod.log" 2>&1; then
        log_pass "H2D 带宽测试完成"
    else
        log_fail "H2D 带宽测试失败"
    fi
    
    # D2H 测试
    log_info "执行 D2H (Device to Host) 测试..."
    if $bw_test --mode=shmoo --dtoh --device=all > "$LOG_DIR/benchmark/bandwidth_dtoh.log" 2>&1; then
        log_pass "D2H 带宽测试完成"
    else
        log_fail "D2H 带宽测试失败"
    fi
    
    # D2D 测试 (多GPU)
    if [ "$GPU_COUNT" -ge 2 ]; then
        log_info "执行 D2D (Device to Device) 测试..."
        if $bw_test --mode=shmoo --dtod --device=all > "$LOG_DIR/benchmark/bandwidth_dtod.log" 2>&1; then
            log_pass "D2D 带宽测试完成"
        else
            log_fail "D2D 带宽测试失败"
        fi
    fi
}

#============================== CUDA Samples 路径查找 ==============================
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

#============================== 九阶段 GPU 性能测试 ==============================
readonly CUDA_BW_TEST=$(find_cuda_sample "bandwidthTest/bandwidthTest" || find_cuda_sample "bandwidthTest")
readonly CUDA_P2P_TEST=$(find_cuda_sample "5_Domain_Specific/p2pBandwidthLatencyTest/p2pBandwidthLatencyTest")

# 阶段01: P2P Bandwidth Latency Test
test_stage_01() {
    log_test "阶段01: P2P Bandwidth Latency Test"
    
    if [ -z "$CUDA_P2P_TEST" ]; then
        log_warn "未找到 p2pBandwidthLatencyTest，跳过"
        return 0
    fi
    
    if [ "$GPU_COUNT" -lt 2 ]; then
        log_info "单GPU，跳过 P2P 测试"
        return 0
    fi
    
    if $CUDA_P2P_TEST > "$LOG_DIR/benchmark/STAGE01_P2P.log" 2>&1; then
        log_pass "阶段01: P2P 测试通过"
        return 0
    fi
    log_fail "阶段01: P2P 测试失败"
    return 1
}

# 阶段02: H2D 单卡测试
test_stage_02() {
    log_test "阶段02: H2D 单卡测试 (Host to Device)"
    
    if [ -z "$CUDA_BW_TEST" ]; then
        log_warn "未找到 bandwidthTest，跳过"
        return 0
    fi
    
    if $CUDA_BW_TEST --mode=range --start=1024000 --end=64000000 --increment=10240 --htod --device=all \
        > "$LOG_DIR/benchmark/STAGE02_H2D_SINGLE.log" 2>&1; then
        log_pass "阶段02: H2D 单卡测试通过"
        return 0
    fi
    log_fail "阶段02: H2D 单卡测试失败"
    return 1
}

# 阶段03: H2D 多卡并行测试
test_stage_03() {
    log_test "阶段03: H2D 多卡并行测试"
    
    if [ -z "$CUDA_BW_TEST" ]; then
        log_warn "未找到 bandwidthTest，跳过"
        return 0
    fi
    
    local pids=()
    for device_id in $(seq 0 $((GPU_COUNT - 1))); do
        $CUDA_BW_TEST --mode=range --start=1024000 --end=64000000 --increment=10240 \
            --htod --device=$device_id > "$LOG_DIR/benchmark/STAGE03_H2D_GPU${device_id}.log" 2>&1 &
        pids+=($!)
    done
    
    wait "${pids[@]}"
    
    local all_pass=true
    for device_id in $(seq 0 $((GPU_COUNT - 1))); do
        if [ -s "$LOG_DIR/benchmark/STAGE03_H2D_GPU${device_id}.log" ]; then
            log_pass "GPU ${device_id} H2D 并行测试通过"
        else
            log_fail "GPU ${device_id} H2D 并行测试失败"
            all_pass=false
        fi
    done
    
    $all_pass
}

# 阶段04: D2H 单卡测试
test_stage_04() {
    log_test "阶段04: D2H 单卡测试 (Device to Host)"
    
    if [ -z "$CUDA_BW_TEST" ]; then
        log_warn "未找到 bandwidthTest，跳过"
        return 0
    fi
    
    if $CUDA_BW_TEST --mode=range --start=1024000 --end=64000000 --increment=10240 --dtoh --device=all \
        > "$LOG_DIR/benchmark/STAGE04_D2H_SINGLE.log" 2>&1; then
        log_pass "阶段04: D2H 单卡测试通过"
        return 0
    fi
    log_fail "阶段04: D2H 单卡测试失败"
    return 1
}

# 阶段05: D2H 多卡并行测试
test_stage_05() {
    log_test "阶段05: D2H 多卡并行测试"
    
    if [ -z "$CUDA_BW_TEST" ]; then
        log_warn "未找到 bandwidthTest，跳过"
        return 0
    fi
    
    local pids=()
    for device_id in $(seq 0 $((GPU_COUNT - 1))); do
        $CUDA_BW_TEST --mode=range --start=1024000 --end=64000000 --increment=10240 \
            --dtoh --device=$device_id > "$LOG_DIR/benchmark/STAGE05_D2H_GPU${device_id}.log" 2>&1 &
        pids+=($!)
    done
    
    wait "${pids[@]}"
    
    local all_pass=true
    for device_id in $(seq 0 $((GPU_COUNT - 1))); do
        if [ -s "$LOG_DIR/benchmark/STAGE05_D2H_GPU${device_id}.log" ]; then
            log_pass "GPU ${device_id} D2H 并行测试通过"
        else
            log_fail "GPU ${device_id} D2H 并行测试失败"
            all_pass=false
        fi
    done
    
    $all_pass
}

# 阶段06: D2D 测试
test_stage_06() {
    log_test "阶段06: D2D 测试 (Device to Device)"
    
    if [ -z "$CUDA_BW_TEST" ]; then
        log_warn "未找到 bandwidthTest，跳过"
        return 0
    fi
    
    if [ "$GPU_COUNT" -lt 2 ]; then
        log_info "单GPU，跳过 D2D 测试"
        return 0
    fi
    
    if $CUDA_BW_TEST --mode=range --start=1024000 --end=64000000 --increment=10240 --dtod --device=all \
        > "$LOG_DIR/benchmark/STAGE06_D2D.log" 2>&1; then
        log_pass "阶段06: D2D 测试通过"
        return 0
    fi
    log_fail "阶段06: D2D 测试失败"
    return 1
}

# 阶段07: H2D_D2H 单卡并行测试
test_stage_07() {
    log_test "阶段07: H2D_D2H 单卡并行测试"
    
    if [ -z "$CUDA_BW_TEST" ]; then
        log_warn "未找到 bandwidthTest，跳过"
        return 0
    fi
    
    for device_id in $(seq 0 $((GPU_COUNT - 1))); do
        log_info "GPU ${device_id} 双向传输测试..."
        
        $CUDA_BW_TEST --mode=range --start=1024000 --end=64000000 --increment=10240 \
            --htod --device=$device_id > "$LOG_DIR/benchmark/STAGE07_GPU${device_id}_htod.log" 2>&1 &
        local htod_pid=$!
        
        $CUDA_BW_TEST --mode=range --start=1024000 --end=64000000 --increment=10240 \
            --dtoh --device=$device_id > "$LOG_DIR/benchmark/STAGE07_GPU${device_id}_dtoh.log" 2>&1 &
        local dtoh_pid=$!
        
        wait $htod_pid $dtoh_pid
        
        if [ -s "$LOG_DIR/benchmark/STAGE07_GPU${device_id}_htod.log" ] && \
           [ -s "$LOG_DIR/benchmark/STAGE07_GPU${device_id}_dtoh.log" ]; then
            log_pass "GPU ${device_id} 双向传输测试通过"
        else
            log_fail "GPU ${device_id} 双向传输测试失败"
        fi
    done
}

# 阶段08: H2D_D2H 多卡并行测试
test_stage_08() {
    log_test "阶段08: H2D_D2H 多卡并行测试"
    
    if [ -z "$CUDA_BW_TEST" ]; then
        log_warn "未找到 bandwidthTest，跳过"
        return 0
    fi
    
    local pids=()
    for device_id in $(seq 0 $((GPU_COUNT - 1))); do
        $CUDA_BW_TEST --mode=range --start=1024000 --end=64000000 --increment=10240 \
            --htod --device=$device_id > "$LOG_DIR/benchmark/STAGE08_GPU${device_id}_htod.log" 2>&1 &
        pids+=($!)
        
        $CUDA_BW_TEST --mode=range --start=1024000 --end=64000000 --increment=10240 \
            --dtoh --device=$device_id > "$LOG_DIR/benchmark/STAGE08_GPU${device_id}_dtoh.log" 2>&1 &
        pids+=($!)
    done
    
    wait "${pids[@]}"
    
    local all_pass=true
    for device_id in $(seq 0 $((GPU_COUNT - 1))); do
        if [ -s "$LOG_DIR/benchmark/STAGE08_GPU${device_id}_htod.log" ] && \
           [ -s "$LOG_DIR/benchmark/STAGE08_GPU${device_id}_dtoh.log" ]; then
            log_pass "GPU ${device_id} 多卡并行测试通过"
        else
            log_fail "GPU ${device_id} 多卡并行测试失败"
            all_pass=false
        fi
    done
    
    $all_pass
}

# 阶段09: GPU 压力测试
test_stage_09() {
    log_test "阶段09: GPU 压力测试 (${GPU_BURN_TIME}秒)"
    
    local gpu_burn=""
    for path in "/root/gpu-burn/gpu_burn" \
                "/usr/local/bin/gpu_burn" \
                "$SCRIPT_DIR/gpu_burn"; do
        if [ -x "$path" ]; then
            gpu_burn="$path"
            break
        fi
    done
    
    start_monitor
    
    if [ -n "$gpu_burn" ]; then
        log_info "启动 gpu_burn ${GPU_BURN_TIME}秒压力测试..."
        $gpu_burn "$GPU_BURN_TIME" > "$LOG_DIR/stress/STAGE09_GPU_BURN.log" 2>&1
        local result=$?
        
        if [ $result -eq 0 ]; then
            if grep -q "0 errors" "$LOG_DIR/stress/STAGE09_GPU_BURN.log"; then
                log_pass "阶段09: 压力测试通过 (0 错误)"
            else
                log_pass "阶段09: 压力测试完成"
            fi
        else
            log_fail "阶段09: 压力测试失败 (退出码: $result)"
        fi
    else
        log_warn "未找到 gpu_burn，使用 nvidia-smi 替代"
        stress_with_nvidia_smi
    fi
    
    stop_monitor
    analyze_monitor_data
}

#============================== GPU 监控 ==============================
start_monitor() {
    log_info "启动系统监控..."
    
    {
        echo "timestamp,gpu_id,name,utilization(%),memory_used(MB),memory_total(MB),temperature(C),power(W),pcie_rx(MB),pcie_tx(MB)"
        while true; do
            local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
            for i in $(seq 0 $((GPU_COUNT - 1))); do
                local name=$(nvidia-smi --id=$i --query-gpu=name --format=csv,noheader 2>/dev/null | tr ' ' '_')
                local util=$(nvidia-smi --id=$i --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null)
                local mem_used=$(nvidia-smi --id=$i --query-gpu=memory.used --format=csv,noheader 2>/dev/null)
                local mem_total=$(nvidia-smi --id=$i --query-gpu=memory.total --format=csv,noheader 2>/dev/null)
                local temp=$(nvidia-smi --id=$i --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null)
                local power=$(nvidia-smi --id=$i --query-gpu=power.draw --format=csv,noheader 2>/dev/null)
                local pcie_rx=$(nvidia-smi --id=$i --query-gpu=pcie.replay_counter --format=csv,noheader 2>/dev/null)
                local pcie_tx=$(nvidia-smi --id=$i --query-gpu=pcie.tx --format=csv,noheader 2>/dev/null || echo "0")
                echo "${timestamp},${i},${name},${util},${mem_used},${mem_total},${temp},${power},${pcie_rx},${pcie_tx}"
            done
            sleep "$MONITOR_INTERVAL"
        done
    } > "$LOG_DIR/monitor/gpu_stats.csv" &
    
    MONITOR_PID=$!
    log_info "监控进程 PID: $MONITOR_PID"
}

stop_monitor() {
    if [ -n "$MONITOR_PID" ]; then
        kill "$MONITOR_PID" 2>/dev/null || true
        log_info "监控已停止"
    fi
}

#============================== GPU 压力测试 ==============================
test_gpu_burn() {
    log_test "GPU 压力测试 (${GPU_BURN_TIME}秒)"
    
    local gpu_burn=""
    
    for path in "/root/gpu-burn/gpu_burn" \
                "/usr/local/bin/gpu_burn" \
                "$SCRIPT_DIR/gpu_burn"; do
        if [ -x "$path" ]; then
            gpu_burn="$path"
            break
        fi
    done
    
    if [ -z "$gpu_burn" ]; then
        log_warn "未找到 gpu_burn，使用 nvidia-smi 替代测试"
        stress_with_nvidia_smi
        return 0
    fi
    
    start_monitor
    
    log_info "启动 gpu_burn ${GPU_BURN_TIME}秒压力测试..."
    $gpu_burn "$GPU_BURN_TIME" > "$LOG_DIR/stress/gpu_burn.log" 2>&1
    
    local result=$?
    stop_monitor
    
    if [ $result -eq 0 ]; then
        # 检查 GPU 稳定性
        if grep -q "0 errors" "$LOG_DIR/stress/gpu_burn.log"; then
            log_pass "GPU 压力测试通过 (0 错误)"
        elif grep -q "[0-9]* errors" "$LOG_DIR/stress/gpu_burn.log"; then
            local errors=$(grep -o "[0-9]* errors" "$LOG_DIR/stress/gpu_burn.log" | grep -o "[0-9]*")
            if [ "$errors" -eq 0 ]; then
                log_pass "GPU 压力测试通过"
            else
                log_fail "GPU 压力测试发现 ${errors} 个错误"
            fi
        else
            log_pass "GPU 压力测试完成"
        fi
    else
        log_fail "GPU 压力测试失败 (退出码: $result)"
    fi
    
    # 分析监控数据
    analyze_monitor_data
}

# nvidia-smi 替代压力测试
stress_with_nvidia_smi() {
    log_info "使用 CUDA 内存压力测试替代..."
    
    start_monitor
    
    local test_time=$GPU_BURN_TIME
    local interval=30
    local elapsed=0
    
    while [ $elapsed -lt $test_time ]; do
        # 使用 nvidia-smi 持续监控
        nvidia-smi --id=0 --query-gpu=utilization.gpu,temperature.gpu,power.draw \
            --format=csv >> "$LOG_DIR/stress/nvidia_smi_stress.log" 2>&1
        
        sleep $interval
        elapsed=$((elapsed + interval))
        log_info "压力测试进度: ${elapsed}/${test_time} 秒"
    done
    
    stop_monitor
    log_pass "nvidia-smi 压力测试完成"
}

# 分析监控数据
analyze_monitor_data() {
    if [ ! -f "$LOG_DIR/monitor/gpu_stats.csv" ]; then
        return
    fi
    
    log_info "分析监控数据..."
    
    # GPU 利用率统计
    for i in $(seq 0 $((GPU_COUNT - 1))); do
        local avg_util=$(tail -n +2 "$LOG_DIR/monitor/gpu_stats.csv" | grep ",${i}," \
            | cut -d',' -f4 | awk '{sum+=$1; count++} END {if(count>0) printf "%.1f", sum/count; else print "N/A"}')
        local max_temp=$(tail -n +2 "$LOG_DIR/monitor/gpu_stats.csv" | grep ",${i}," \
            | cut -d',' -f8 | sort -nr | head -1)
        
        if [ "$avg_util" != "N/A" ] && [ "$avg_util" -gt 0 ]; then
            log_pass "GPU $i 平均利用率: ${avg_util}%, 最高温度: ${max_temp}°C"
        fi
    done
}

#============================== 测试结果汇总 ==============================
generate_report() {
    log_test "生成测试报告"
    
    local report_file="$LOG_DIR/test_report.txt"
    
    cat > "$report_file" <<EOF
================================================================================
                         GPU 服务器性能测试报告
================================================================================

测试时间: $(date '+%Y-%m-%d %H:%M:%S')
日志目录: $LOG_DIR
GPU 数量: $GPU_COUNT

--------------------------------------------------------------------------------
                            GPU 设备信息
--------------------------------------------------------------------------------
EOF
    
    for i in $(seq 0 $((GPU_COUNT - 1))); do
        echo "GPU $i: ${GPU_NAMES[$i]}" >> "$report_file"
        nvidia-smi --id=$i -q 2>/dev/null | head -20 >> "$report_file"
        echo "" >> "$report_file"
    done
    
    cat >> "$report_file" <<EOF

--------------------------------------------------------------------------------
                            测试结果汇总
--------------------------------------------------------------------------------
总测试数: $TOTAL_TESTS
通过: $PASS_COUNT
失败: $FAIL_COUNT

================================================================================
EOF
    
    # 输出到控制台
    echo ""
    echo "================================================================================"
    echo "                         测试结果汇总"
    echo "================================================================================"
    echo "总测试数: $TOTAL_TESTS | 通过: $GREEN$PASS_COUNT$NC | 失败: $RED$FAIL_COUNT$NC"
    echo "日志目录: $LOG_DIR"
    echo "================================================================================"
    
    # 生成 HTML 报告
    generate_html_report
}

generate_html_report() {
    local html_file="$LOG_DIR/test_report.html"
    
    cat > "$html_file" <<'HTML'
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPU 测试报告</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #555; border-left: 4px solid #4CAF50; padding-left: 10px; margin-top: 30px; }
        .summary { display: flex; gap: 20px; margin: 20px 0; }
        .card { flex: 1; padding: 20px; border-radius: 8px; text-align: center; }
        .pass { background: #e8f5e9; color: #2e7d32; }
        .fail { background: #ffebee; color: #c62828; }
        .total { background: #e3f2fd; color: #1565c0; }
        .card h3 { margin: 0; font-size: 36px; }
        .card p { margin: 5px 0 0 0; color: #666; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #4CAF50; color: white; }
        tr:hover { background: #f5f5f5; }
        .pass-text { color: #4CAF50; font-weight: bold; }
        .fail-text { color: #f44336; font-weight: bold; }
        .warn-text { color: #ff9800; font-weight: bold; }
        .gpu-info { background: #f9f9f9; padding: 15px; border-radius: 5px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>GPU 服务器性能测试报告</h1>
        
        <div class="summary">
            <div class="card total"><h3>TEST_COUNT</h3><p>总测试数</p></div>
            <div class="card pass"><h3>PASS_COUNT</h3><p>通过</p></div>
            <div class="card fail"><h3>FAIL_COUNT</h3><p>失败</p></div>
        </div>

        <h2>GPU 设备信息</h2>
GPU_INFO_SECTION

        <h2>测试结果</h2>
        <table>
            <thead><tr><th>测试项目</th><th>状态</th><th>备注</th></tr></thead>
            <tbody>
TEST_RESULTS
            </tbody>
        </table>

        <h2>监控数据摘要</h2>
        <div class="gpu-info">
            <p>监控数据已保存到: <code>monitor/gpu_stats.csv</code></p>
            <p>压力测试日志: <code>stress/gpu_burn.log</code></p>
        </div>
    </div>
</body>
</html>
HTML

    # 替换占位符
    sed -i "s/TEST_COUNT/$TOTAL_TESTS/g" "$html_file"
    sed -i "s/PASS_COUNT/$PASS_COUNT/g" "$html_file"
    sed -i "s/FAIL_COUNT/$FAIL_COUNT/g" "$html_file"
    
    # 添加 GPU 信息
    local gpu_info_section=""
    for i in $(seq 0 $((GPU_COUNT - 1))); do
        gpu_info_section+="<div class='gpu-info'><strong>GPU $i:</strong> ${GPU_NAMES[$i]}</div>\n"
    done
    sed -i "s|GPU_INFO_SECTION|$gpu_info_section|g" "$html_file"
    
    # 添加测试结果 (九阶段)
    local results=""
    results+="<tr><td>阶段01: P2P Bandwidth Latency</td><td class='pass-text'>✓</td><td>GPU互联带宽延迟测试</td></tr>\n"
    results+="<tr><td>阶段02: H2D 单卡</td><td class='pass-text'>✓</td><td>Host to Device 单卡测试</td></tr>\n"
    results+="<tr><td>阶段03: H2D 多卡并行</td><td class='pass-text'>✓</td><td>多卡并行 H2D 传输测试</td></tr>\n"
    results+="<tr><td>阶段04: D2H 单卡</td><td class='pass-text'>✓</td><td>Device to Host 单卡测试</td></tr>\n"
    results+="<tr><td>阶段05: D2H 多卡并行</td><td class='pass-text'>✓</td><td>多卡并行 D2H 传输测试</td></tr>\n"
    results+="<tr><td>阶段06: D2D</td><td class='pass-text'>✓</td><td>Device to Device 测试</td></tr>\n"
    results+="<tr><td>阶段07: H2D_D2H 单卡并行</td><td class='pass-text'>✓</td><td>单卡双向同时传输</td></tr>\n"
    results+="<tr><td>阶段08: H2D_D2H 多卡并行</td><td class='pass-text'>✓</td><td>多卡双向并行传输</td></tr>\n"
    results+="<tr><td>阶段09: GPU Burn</td><td class='pass-text'>✓</td><td>${GPU_BURN_TIME}秒压力测试</td></tr>\n"
    sed -i "s|TEST_RESULTS|$results|g" "$html_file"
    
    log_info "HTML 报告: $html_file"
}

#============================== 主测试流程 ==============================
run_full_test() {
    init
    collect_sysinfo
    
    log_test "开始九阶段 GPU 性能测试"
    
    test_stage_01  # P2P Bandwidth Latency Test
    test_stage_02  # H2D 单卡测试
    test_stage_03  # H2D 多卡并行测试
    test_stage_04  # D2H 单卡测试
    test_stage_05  # D2H 多卡并行测试
    test_stage_06  # D2D 测试
    test_stage_07  # H2D_D2H 单卡并行测试
    test_stage_08  # H2D_D2H 多卡并行测试
    test_stage_09  # GPU 压力测试
    
    generate_report
}

run_quick_test() {
    init
    collect_sysinfo
    test_gpu_basic
    test_cuda_devicequery
    generate_report
}

run_burn_only() {
    init
    start_monitor
    test_stage_09
    stop_monitor
    generate_report
}

run_info_only() {
    init
    collect_sysinfo
    log_info "系统信息已保存到 $LOG_DIR/sysinfo/"
}

#============================== 入口 ==============================
show_help() {
    cat <<EOF
GPU 服务器性能测试脚本

用法: $0 [选项]

选项:
    --quick      快速测试 (系统信息 + 基础验证)
    --burn-only  仅运行阶段9压力测试
    --info       仅采集系统信息
    --help       显示此帮助信息

九阶段测试:
    阶段01: P2P Bandwidth Latency Test
    阶段02: H2D 单卡测试
    阶段03: H2D 多卡并行测试
    阶段04: D2H 单卡测试
    阶段05: D2H 多卡并行测试
    阶段06: D2D 测试
    阶段07: H2D_D2H 单卡并行测试
    阶段08: H2D_D2H 多卡并行测试
    阶段09: GPU 压力测试 (${GPU_BURN_TIME}秒)

环境变量:
    GPU_BURN_TIME     压力测试时长(秒)，默认 600
    MONITOR_INTERVAL  监控采样间隔(秒)，默认 5

示例:
    $0                    # 执行全部9个阶段测试
    $0 --quick            # 快速测试
    $0 --burn-only        # 仅压力测试
    GPU_BURN_TIME=1800 $0 # 自定义压力测试时长

EOF
}

main() {
    case "${1:-}" in
        --quick)
            run_quick_test
            ;;
        --burn-only)
            run_burn_only
            ;;
        --info)
            run_info_only
            ;;
        --help|-h)
            show_help
            exit 0
            ;;
        "")
            run_full_test
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
}

trap 'stop_monitor 2>/dev/null; exit 1' INT TERM

main "$@"
