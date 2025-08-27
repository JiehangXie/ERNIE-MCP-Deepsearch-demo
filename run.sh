#!/bin/bash

# 检查是否安装了uv
if ! command -v uv &> /dev/null; then
    echo "uv未安装，正在安装..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

# 安装依赖
echo "正在安装依赖..."
uv sync

# 运行项目
echo "启动MCP客户端..."
uv run python main.py