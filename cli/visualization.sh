#!/bin/bash
# 终端可视化组件库

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # 颜色重置

# 步骤提示函数
print_step() {
    echo -e "${BLUE}==>[$(date +%H:%M:%S)] 步骤 $1: $2${NC}"
}

# 进度条函数
show_progress() {
    local duration=${1:-5}
    local width=${2:-40}
    
    echo -ne "${YELLOW}["
    for ((i=0; i<width; i++)); do
        sleep $(bc <<< "scale=3; $duration/$width")
        echo -n "▋"
    done
    echo -e "]${NC}\n"
}

# 完成状态标记
show_complete() {
    echo -e "${GREEN}✔ $1 完成${NC}\n"
}

# 错误提示
show_error() {
    echo -e "${RED}✖ 错误: $1${NC}"
    exit 1
}