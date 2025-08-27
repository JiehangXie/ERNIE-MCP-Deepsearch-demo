# ERNIE-MCP-Deepsearch-demo

这是一个基于文心大模型（ERNIE）和MCP (Model Context Protocol) 的混合检索增强项目，支持连接多种MCP服务器并进行智能对话。

## 功能特性

- 🔌 支持多种MCP服务器连接（SSE、stdio）
- 🤖 集成多代理系统，支持任务分解和执行
- 🔍 支持百度搜索、ArXiv论文搜索、B站视频搜索等
- 🎨 美观的命令行界面（基于Rich）
- ⚡ 异步处理，高性能

## 快速开始

### 环境配置

1. **复制环境变量文件**：
   ```bash
   cp .env.example .env
   ```

2. **编辑.env文件**，填入你的API密钥：
   ```env
   ERNIE_BASE_URL=https://aistudio.baidu.com/llm/lmapi/v3
   ERNIE_CHAT_MODEL=ernie-4.5-turbo-128k
   ERNIE_THINKING_MODEL=ernie-x1-turbo-32k
   ERNIE_APIKEY=your_api_key_here
   MCP_CONFIG_PATH=config.json
   BAIDU_SEARCH_API_KEY=your_baidu_search_api_key
   ```

3. **配置MCP服务器**：
   编辑`config.json`文件，启用或禁用需要的服务器。

4. **安装Node.js环境 （可选，用于部分MCP服务器）**
如果你需要使用B站视频搜索或DuckDuckGo网络搜索等基于Node.js的MCP服务器，请安装Node.js：  
macOS（使用Homebrew） ：
```bash
brew install node
```
Linux（Ubuntu/Debian） ：
```bash
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt-get install -y nodejs
```
Windows ：
下载并安装Node.js LTS版本：https://nodejs.org/en/download/  

安装完成后，验证安装：
```bash
node -v
npm -v
```
💡 提示 ：Node.js版本建议使用18.x或更高版本以确保最佳兼容性。

### 使用uv（推荐）

1. **安装uv**（如果尚未安装）：
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **克隆项目并进入目录**：
   ```bash
   git clone https://github.com/JiehangXie/ERNIE-MCP-Deepsearch-demo.git
   cd ERNIE-MCP-Deepsearch-demo
   ```

3. **一键运行**：
   ```bash
   chmod +x run.sh
   ./run.sh
   ```

   或者手动运行：
   ```bash
   # 安装依赖
   uv sync
   
   # 运行项目
   uv run python main.py
   ```
