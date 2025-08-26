import asyncio
import os
from dotenv import load_dotenv
from client import MCPClient

load_dotenv()

async def main():
    # 获取配置文件路径
    config_path = os.getenv("MCP_CONFIG_PATH", "config.json")
    
    # 创建MCP客户端
    client = MCPClient(config_path=config_path)
    
    try:
        # 启动聊天循环
        await client.chat_loop()
    finally:
        # 清理连接
        await client.cleanup()
        print("\n已断开所有服务器连接")

if __name__ == "__main__":
    asyncio.run(main())