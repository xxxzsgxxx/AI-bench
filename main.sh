#!/bin/bash
# 模块化主控脚本
set -e

# 加载参数解析模块
source "$(dirname "$0")/cli/cli_parser.sh"

# 初始化日志系统
init_logging() {
    LOG_BASE="../sysinfo_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "${LOG_BASE}/runtime_logs"
}

# 执行环境初始化
execute_init() {
    echo "=== 系统环境初始化 ==="
    ./install.sh | tee "${LOG_BASE}/runtime_logs/init.log"
}

# 执行系统信息采集
execute_sysinfo() {
    echo "正在执行系统信息采集..."
# 统一硬件检测调用方式
lshw -short 2>&1 | tee -a ${LOG_BASE}/runtime_logs/hardware/system_hardware.log
sudo dmidecode -t bios 2>&1 | tee -a ${LOG_BASE}/runtime_logs/hardware/bios_info.log || echo "警告: 需要root权限获取BIOS信息"
    ./sysinfo/sysinfo_collector.sh | tee "${LOG_BASE}/runtime_logs/sysinfo.log"
}

# 执行基准测试
execute_benchmark() {
    echo "\n=== 启动综合性能测试 ==="
    ./benchmark_cpu_mem_disk.sh
    ./benchmark_ai_frameworks.sh | tee "${LOG_BASE}/runtime_logs/ai_benchmark.log"
    
    echo "\n=== 启动资源监控 ==="
    ./monitor_resources.sh &
    MONITOR_PID=$!
    trap "kill $MONITOR_PID 2>/dev/null" EXIT
}

# 生成测试报告
generate_report() {
    echo "\n=== 生成最终测试报告 ==="
    ../report_generator/report_generator.sh "${LOG_BASE}"
}

main() {
    parse_arguments "$@"
    init_logging

    [[ $INIT_MODE ]] && execute_init
    [[ $SYSINFO_MODE ]] && execute_sysinfo
    [[ $BENCH_MODE ]] && execute_benchmark
    [[ $REPORT_MODE ]] && generate_report

    # 全模式执行
    if [[ $FULL_MODE ]]; then
        execute_init
        execute_sysinfo
        execute_benchmark
        generate_report
    fi

    echo "\n操作完成！日志目录: ${LOG_BASE}"
    wait
}

main "$@"