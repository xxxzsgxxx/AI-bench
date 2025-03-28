# AI Benchmark Toolkit

# 版本控制

## Git协作规范

### 分支管理
1. 主分支（main）保持随时可部署状态
2. 功能开发使用特性分支：
```bash
git checkout -b feature/your-feature-name
```
3. 使用rebase保持提交历史整洁

### 提交规范
- 提交信息格式：
```
[类型] 简要描述

详细说明（可选）
```
- 常用类型：feat/fix/docs/style/refactor/test

### 冲突处理流程
1. 保持本地分支与main分支同步
```bash
git checkout main
git pull origin main
git checkout your-branch
git rebase main
```
2. 解决冲突后继续rebase
3. 强制推送特性分支（仅限个人分支）
```bash
git push -f origin your-branch
```

跨平台系统性能分析与验证工具集

## 功能特性
- 硬件信息采集
- 运行时环境验证
- 性能基准测试
- 多格式报告生成

## 快速开始
```bash
./install.sh
./main.sh --module=hardware
```