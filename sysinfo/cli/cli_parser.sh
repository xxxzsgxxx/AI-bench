#!/bin/bash
# 命令行参数解析模块

parse_arguments() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --init)
                INIT_MODE=true
                shift
                ;;
            --sysinfo)
                SYSINFO_MODE=true
                shift
                ;;
            --bench)
                BENCH_MODE=true
                shift
                ;;
            --report)
                REPORT_MODE=true
                shift
                ;;
            --all)
                FULL_MODE=true
                shift
                ;;
            -*)
                echo "未知参数: $1"
                exit 1
                ;;
            *)
                shift
                ;;
        esac
    done

    # 自动关联依赖步骤
    [[ $BENCH_MODE ]] && INIT_MODE=true
    [[ $REPORT_MODE ]] && SYSINFO_MODE=true
}

export -f parse_arguments