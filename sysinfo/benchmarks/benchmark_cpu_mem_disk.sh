#!/bin/bash
# 综合性能基准测试模块
set -e

LOG_DIR="../sysinfo_$(date +%Y%m%d_%H%M%S)/benchmarks"
mkdir -p ${LOG_DIR}/{cpu,memory,storage}

# CPU基准测试（多核浮点运算）
benchmark_cpu() {
    echo "[1/3] 开始CPU压力测试..."
    stress-ng --cpu $(nproc) --cpu-method fft --metrics-brief -t 1m \
        | tee ${LOG_DIR}/cpu/stress-ng.log
    
    # 解析关键指标
    grep 'cpu ' ${LOG_DIR}/cpu/stress-ng.log | awk '{print "CPU平均负载:",$4"%"}' > ${LOG_DIR}/cpu/summary.log
}

# 内存带宽测试
benchmark_memory() {
    echo "[2/3] 开始内存带宽测试..."
    # 写入速度测试
    dd if=/dev/zero of=/tmp/test bs=1G count=1 oflag=direct 2> ${LOG_DIR}/memory/dd_write.log
    
    # 内存压力测试
    stress-ng --vm 2 --vm-bytes 4G --vm-method rowhammer -t 30s \
        | tee ${LOG_DIR}/memory/stress-ng.log
    
    # 清理临时文件
    rm -f /tmp/test
}

# 存储性能测试
benchmark_storage() {
    echo "[3/3] 开始存储IO测试..."
    # 顺序读写测试
    fio --name=seq_read --rw=read --bs=1M --size=1G --output=${LOG_DIR}/storage/seq_read.log
    fio --name=seq_write --rw=write --bs=1M --size=1G --output=${LOG_DIR}/storage/seq_write.log
    
    # 随机IO测试
    fio --name=rand_rw --rw=randrw --bs=4k --size=1G --runtime=60 --iodepth=64 --ioengine=libaio --direct=1 --group_reporting \
        | tee ${LOG_DIR}/storage/random_io.log
}

main() {
    benchmark_cpu
    benchmark_memory
    benchmark_storage
    echo "测试完成！日志目录：${LOG_DIR}"
}

main "$@"