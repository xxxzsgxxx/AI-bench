#!/bin/bash
# Git工作流程示例
set -e

echo '1. 初始化本地仓库'
git init
echo '2. 创建并切换到新分支'
git checkout -b feature/sample-change

echo '3. 模拟文件修改'
echo "# Sample" > README.md
git add .
git commit -m "[feat] 添加示例文件"

echo '4. 解决分支冲突场景'
git checkout main 2>/dev/null || git checkout -b main
git merge feature/sample-change --no-ff -m "合并示例分支" || {
    echo '5. 手动解决冲突后执行：'
    echo '   git add .'
    echo '   git commit -m "解决合并冲突"'
    echo '   git push origin main --force'
}

echo '6. 清理临时分支'
git branch -D feature/sample-change

echo '操作说明：\n- 步骤4演示合并冲突处理流程\n- 使用--force推送前请确保分支权限\n- 实际使用时移除--force参数更安全'