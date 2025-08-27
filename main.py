import asyncio
import os
import signal
import sys
from dotenv import load_dotenv
from client import MCPClient

load_dotenv()

def signal_handler(signum, frame):
    """处理信号中断"""
    print("\n\n🛑 检测到中断信号，正在安全退出...")
    sys.exit(0)

async def main():
    # 注册信号处理器 (支持Mac/Linux/Windows)
    signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
    if hasattr(signal, 'SIGTERM'):
        signal.signal(signal.SIGTERM, signal_handler)  # 终止信号
    
    # 获取配置文件路径
    config_path = os.getenv("MCP_CONFIG_PATH", "config.json")
    
    # 创建MCP客户端
    client = MCPClient(config_path=config_path)
    
    try:
        # 启动聊天循环
        await client.chat_loop()
    except KeyboardInterrupt:
        print("\n\n🛑 程序被用户中断")
    finally:
        # 清理连接
        await client.cleanup()
        print("\n已断开所有服务器连接")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 程序已安全退出")
        sys.exit(0)