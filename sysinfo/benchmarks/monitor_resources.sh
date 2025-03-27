#!/bin/bash
# 资源监控模块
set -e

LOG_DIR="../sysinfo_$(date +%Y%m%d_%H%M%S)/monitor"
mkdir -p ${LOG_DIR}

monitor_gpu() {
    echo "[GPU监控] 开始记录GPU指标..."
    nvidia-smi --query-gpu=timestamp,temperature.gpu,utilization.gpu,memory.used,power.draw --format=csv -l 1 -f ${LOG_DIR}/gpu_metrics.csv
}

monitor_system() {
    echo "[系统监控] 开始记录CPU/内存指标..."
    sar -u -r -d 1 60 > ${LOG_DIR}/system_metrics.log &
    vmstat -n 1 60 > ${LOG_DIR}/vmstat.log &
}

generate_report() {
    echo "生成监控报告..."
    echo "### 资源监控报告 ###" > ${LOG_DIR}/summary.log
    date '+%Y-%m-%d %H:%M:%S' >> ${LOG_DIR}/summary.log
    
    # GPU最高温度
    awk -F',' 'NR>1 {print $2}' ${LOG_DIR}/gpu_metrics.csv | sort -nr | head -1 | \
        xargs -I{} echo "GPU最高温度: {}°C" >> ${LOG_DIR}/summary.log
    
    # 平均CPU使用率
    grep -v "%idle" ${LOG_DIR}/system_metrics.log | awk -F' ' '{print 100 - $NF}' | \
        awk '{sum+=$1} END {printf "平均CPU使用率: %.1f%%\n", sum/NR}' >> ${LOG_DIR}/summary.log
}

main() {
    monitor_gpu &
    monitor_system
    wait
    generate_report
    echo "监控完成！数据保存在：${LOG_DIR}"
}

main "$@"