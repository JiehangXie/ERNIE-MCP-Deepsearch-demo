import asyncio
import json
import os
import re
from typing import Optional, List, Dict, Any
from contextlib import AsyncExitStack
import time
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

from openai import AsyncOpenAI
from prompt import TASK_DECOMPOSITION_PROMPT, TOOL_EXECUTION_PROMPT
from json_repair import repair_json
from rich import print
from mcp_agent import MCPAgent
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

class MCPClient:
    def __init__(self, config_path: str = "config.json"):
        # Initialize session and client objects
        self.sessions: Dict[str, ClientSession] = {}
        self.exit_stack = AsyncExitStack()
        self.openai = AsyncOpenAI(api_key=os.getenv("ERNIE_APIKEY"), 
                        base_url=os.getenv("ERNIE_BASE_URL"))
        self.config_path = config_path
        self.config = self.load_config()
        self._streams_contexts = {}
        self._session_contexts = {}
        # 添加console初始化
        self.console = Console()
        # 添加agent初始化
        self.agent = None
        self.agent: Optional[MCPAgent] = None
        
    def load_config(self) -> dict:
        """Load configuration from JSON file and replace environment variables"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_content = f.read()
                
            # Replace environment variables in the format ${VAR_NAME}
            def replace_env_var(match):
                var_name = match.group(1)
                return os.getenv(var_name, match.group(0))  # Return original if env var not found
            
            config_content = re.sub(r'\$\{([^}]+)\}', replace_env_var, config_content)
            
            return json.loads(config_content)
        except FileNotFoundError:
            print(f"配置文件 {self.config_path} 未找到，使用默认配置")
            return {
                "mcpServers": {},
                "connection_settings": {
                    "timeout": 30,
                    "retry_attempts": 3,
                    "retry_delay": 5
                }
            }
        except json.JSONDecodeError as e:
            print(f"配置文件格式错误: {e}")
            return {"mcpServers": {}, "connection_settings": {}}

    def get_server_configs(self) -> Dict[str, dict]:
        """Get all server configurations"""
        return self.config.get("mcpServers", {})

    def is_server_enabled(self, server_config: dict) -> bool:
        """Check if a server is enabled"""
        return server_config.get("enabled", True)

    def get_server_type(self, server_config: dict) -> str:
        """Determine server type based on configuration"""
        if "type" in server_config:
            return server_config["type"].lower()
        elif "command" in server_config:
            return "stdio"
        elif "url" in server_config:
            return "sse"
        else:
            return "unknown"

    async def connect_to_sse_server(self, server_url: str, server_name: str = "default"):
        """Connect to an MCP server running with SSE transport"""
        try:
            # Store the context managers so they stay alive
            self._streams_contexts[server_name] = sse_client(url=server_url)
            streams = await self._streams_contexts[server_name].__aenter__()

            self._session_contexts[server_name] = ClientSession(*streams)
            session = await self._session_contexts[server_name].__aenter__()
            
            # Initialize
            await session.initialize()
            
            # Store the session
            self.sessions[server_name] = session

            # List available tools to verify connection
            print(f"已初始化SSE客户端连接到服务器: {server_name}")
            response = await session.list_tools()
            tools = response.tools
            print(f"已连接到服务器 {server_name}，可用工具: {[tool.name for tool in tools]}")
            
            return True
        except Exception as e:
            print(f"连接SSE服务器 {server_name} ({server_url}) 失败: {e}")
            return False

    async def connect_to_stdio_server(self, command: str, args: List[str], server_name: str = "default"):
        """Connect to an MCP server running with stdio transport"""
        try:
            # Create StdioServerParameters object
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env=None
            )
            
            # Store the context managers so they stay alive
            self._streams_contexts[server_name] = stdio_client(server_params)
            streams = await self._streams_contexts[server_name].__aenter__()

            self._session_contexts[server_name] = ClientSession(*streams)
            session = await self._session_contexts[server_name].__aenter__()
            
            # Initialize
            await session.initialize()
            
            # Store the session
            self.sessions[server_name] = session

            # List available tools to verify connection
            print(f"已初始化stdio客户端连接到服务器: {server_name}")
            response = await session.list_tools()
            tools = response.tools
            print(f"已连接到服务器 {server_name}，可用工具: {[tool.name for tool in tools]}")
            
            return True
        except Exception as e:
            print(f"连接stdio服务器 {server_name} (命令: {command} {' '.join(args)}) 失败: {e}")
            return False

    async def connect_to_server(self, server_name: str, server_config: dict):
        """Connect to a server based on its configuration"""
        server_type = self.get_server_type(server_config)
        
        if server_type == "sse":
            server_url = server_config["url"]
            return await self.connect_to_sse_server(server_url, server_name)
        elif server_type == "stdio":
            command = server_config["command"]
            args = server_config.get("args", [])
            return await self.connect_to_stdio_server(command, args, server_name)
        else:
            print(f"不支持的服务器类型: {server_type} (服务器: {server_name})")
            return False

    async def connect_to_all_enabled_servers(self):
        """Connect to all enabled servers from config"""
        server_configs = self.get_server_configs()
        enabled_servers = {name: config for name, config in server_configs.items() 
                          if self.is_server_enabled(config)}
        
        if not enabled_servers:
            print("配置文件中没有启用的服务器")
            return
            
        print(f"正在连接到 {len(enabled_servers)} 个启用的服务器...")
        
        connection_tasks = []
        for server_name, server_config in enabled_servers.items():
            task = self.connect_to_server(server_name, server_config)
            connection_tasks.append(task)
            
        results = await asyncio.gather(*connection_tasks, return_exceptions=True)
        
        successful_connections = sum(1 for result in results if result is True)
        print(f"\n成功连接到 {successful_connections}/{len(enabled_servers)} 个服务器")

    async def cleanup(self):
        """Properly clean up all sessions and streams"""
        for server_name in list(self._session_contexts.keys()):
            try:
                await self._session_contexts[server_name].__aexit__(None, None, None)
                await self._streams_contexts[server_name].__aexit__(None, None, None)
            except Exception as e:
                print(f"清理服务器 {server_name} 连接时出错: {e}")
        
        self.sessions.clear()
        self._session_contexts.clear()
        self._streams_contexts.clear()

    async def process_query(self, query: str, preferred_server: str = None) -> str:
        """Process a query using OpenAI API and available tools with task decomposition"""
        if not self.sessions:
            return "错误: 没有可用的服务器连接"
        
        # Step 1: Collect all available tools from all servers with server info
        print("[bold magenta]=== 收集所有服务器工具信息 ===[/bold magenta]")
        all_tools_with_server = []
        
        for server_name, session in self.sessions.items():
            try:
                response = await session.list_tools()
                server_tools = [{ 
                    "server": server_name,
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                } for tool in response.tools]
                
                all_tools_with_server.extend(server_tools)
                print(f"服务器 {server_name}: {len(server_tools)} 个工具")
                
            except Exception as e:
                print(f"获取服务器 {server_name} 工具失败: {str(e)}")
        
        # Step 2: Decompose user query into steps using all available tools
        print("[bold magenta]=== 正在分解任务 ===[/bold magenta]")
        decomposition_result = await self._decompose_query(query, all_tools_with_server)
        
        if not decomposition_result:
            return "错误: 无法分解用户问题"
            
        # 使用rich将任务分解结果打印出来
        print("[bold magenta]=== 执行步骤 ===[/bold magenta]")
        for step in decomposition_result.get("steps", []):
            server_info = f" (服务器: {step.get('server_name', '未指定')})"
            print(f"[green]{step['step_number']}. {step['description']}{server_info}[/green]")
        
        # Step 3: Initialize variables for execution
        execution_results = []
        previous_results = []  # Store results from previous steps
        
        # Step 4: Execute steps sequentially using specified servers
        for step in decomposition_result.get("steps", []):
            print(f"\n执行步骤 {step['step_number']}: {step['description']}")
            
            try:
                tool_name = step["tool_name"]
                server_name = step.get("server_name")
                
                # Use the dedicated method to prepare parameters
                tool_params = self._prepare_step_params(step, previous_results)
                
                # Validate server exists
                if not server_name or server_name not in self.sessions:
                    # Fallback: find any server that has this tool
                    available_server = None
                    for srv_name, session in self.sessions.items():
                        response = await session.list_tools()
                        if any(tool.name == tool_name for tool in response.tools):
                            available_server = srv_name
                            break
                    
                    if not available_server:
                        raise Exception(f"工具 {tool_name} 在任何服务器上都不可用")
                    
                    server_name = available_server
                    print(f"警告: 使用备选服务器 {server_name}")
                
                print(f"使用服务器 {server_name} 执行工具 {tool_name}")
                print(f"参数: {json.dumps(tool_params, ensure_ascii=False, indent=2)}")
                
                # Execute tool call on specified server
                session = self.sessions[server_name]
                result = await session.call_tool(tool_name, tool_params)
                
                step_result = {
                    "step": step,
                    "result": result.content[0].text,
                    "server": server_name,
                    "success": True
                }
                
                execution_results.append(step_result)
                previous_results.append(step_result)  # Add to previous results for next steps
                
                print(f"步骤 {step['step_number']} 在服务器 {server_name} 上执行成功")
                print(f"结果: {result.content[0].text[:200]}..." if len(result.content[0].text) > 200 else f"结果: {result.content[0].text}")
                
            except Exception as e:
                print(f"步骤 {step['step_number']} 执行失败: {str(e)}")
                error_result = {
                    "step": step,
                    "error": str(e),
                    "server": step.get("server_name", "未知"),
                    "success": False
                }
                execution_results.append(error_result)
                # Don't add failed results to previous_results to avoid propagating errors
        
        # Step 5: Generate final response
        final_response = await self._generate_final_response(query, execution_results)
        
        return final_response
    
    async def _decompose_query(self, query: str, available_tools: List[dict]) -> Optional[dict]:
        """Decompose user query into executable steps"""
        try:
            # Format available tools for prompt with server information
            tools_description = "\n".join([
                f"- 服务器: {tool['server']}\n"
                f"  工具名称: {tool['name']}\n"
                f"  描述: {tool['description']}\n"
                f"  参数定义: {json.dumps(tool['parameters'], ensure_ascii=False, indent=2)}\n"
                for tool in available_tools
            ])
            
            # Create decomposition prompt
            prompt = TASK_DECOMPOSITION_PROMPT.format(
                user_query=query,
                available_tools=tools_description
            )
            
            # Call OpenAI API for decomposition
            completion = await self.openai.chat.completions.create(
                model=os.getenv("ERNIE_THINKING_MODEL"),
                max_tokens=4096,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.6
            )
            
            # Parse JSON response
            response_text = completion.choices[0].message.content
            response_text = repair_json(response_text)
            
            return json.loads(response_text)
            
        except Exception as e:
            print(f"任务分解失败: {str(e)}")
            return None
    
    async def _generate_final_response(self, original_query: str, execution_results: List[dict]) -> str:
        """Generate final response based on execution results"""
        try:
            # Format execution results
            results_text = "\n".join([
                f"步骤 {result['step']['step_number']}: {result['step']['description']}\n"
                f"服务器: {result.get('server', '未知')}\n"
                f"结果: {result.get('result', result.get('error', '未知错误'))}\n"
                f"状态: {'成功' if result['success'] else '失败'}\n"
                for result in execution_results
            ])
            
            # Create final response prompt
            prompt = TOOL_EXECUTION_PROMPT.format(
                user_query=original_query,
                execution_results=results_text
            )
            
            # Call OpenAI API for final response
            completion = await self.openai.chat.completions.create(
                model=os.getenv("ERNIE_CHAT_MODEL"),
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            print(f"生成最终回复失败: {str(e)}")
            # Fallback: return execution results directly
            return "\n".join([
                f"步骤 {result['step']['step_number']}: {result['step']['description']} - {'成功' if result['success'] else '失败'} (服务器: {result.get('server', '未知')})"
                for result in execution_results
            ])

    def list_servers(self):
        """List all configured servers and their status"""
        print("\n配置的服务器:")
        server_configs = self.get_server_configs()
        
        for server_name, server_config in server_configs.items():
            status = "已连接" if server_name in self.sessions else "未连接"
            enabled = "启用" if self.is_server_enabled(server_config) else "禁用"
            server_type = self.get_server_type(server_config).upper()
            
            if server_type == "SSE":
                endpoint = server_config.get("url", "未知")
            elif server_type == "STDIO":
                command = server_config.get("command", "未知")
                args = server_config.get("args", [])
                endpoint = f"{command} {' '.join(args)}"
            else:
                endpoint = "未知类型"
                
            print(f"  - {server_name} ({server_type}): {endpoint} ({enabled}, {status})")
            if server_config.get("description"):
                print(f"    描述: {server_config['description']}")

    async def initialize_agent(self):
        """初始化智能Agent"""
        agent_config = {
            "max_reflection_cycles": 3,
            "enable_dynamic_replanning": True,
            "thinking_model": os.getenv("ERNIE_THINKING_MODEL"),
            "chat_model": os.getenv("ERNIE_CHAT_MODEL")
        }
        
        self.agent = MCPAgent("MCP_Agent", agent_config, self)
        print("智能Agent已初始化")
    
    async def process_query_with_agent(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """使用智能Agent处理查询"""
        if not self.agent:
            await self.initialize_agent()
        
        return await self.agent.process_query(query, context)
    
    async def chat_loop(self):
        """运行交互式聊天循环，增强版"""
        self.console.print(Panel.fit(
            "🤖 [bold blue]MCP智能客户端[/bold blue]\n\n" +
            "支持两种模式:\n" +
            "• [green]智能模式[/green]: 多步规划、思考、执行、反思\n" +
            "• [yellow]简单模式[/yellow]: 直接任务分解执行\n\n" +
            "输入 'help' 查看帮助信息\n" +
            "[dim]按 Ctrl+C 可随时取消当前操作，连续按两次退出程序[/dim]",
            title="欢迎使用",
            border_style="blue"
        ))
        
        # 连接到所有启用的服务器
        await self.connect_to_all_enabled_servers()
        
        # 初始化Agent
        await self.initialize_agent()
        
        # 显示服务器状态
        self.list_servers()
        
        consecutive_interrupts = 0  # 记录连续中断次数
        
        while True:
            try:
                consecutive_interrupts = 0  # 重置中断计数
                self.console.print("\n" + "="*60)
                query = Prompt.ask(
                    "\n[bold cyan]请输入您的查询[/bold cyan] [dim](Ctrl+C取消)[/dim]",
                ).strip()
                
                if query.lower() in ['quit', 'exit', '退出', 'q']:
                    self.console.print("[yellow]👋 再见![/yellow]")
                    break
                elif query.lower() in ['help', '帮助', 'h']:
                    self._show_help()
                    continue
                elif query.lower() in ['servers', '服务器', 's']:
                    self.list_servers()
                    continue
                elif query.lower() in ['clear', '清屏', 'c']:
                    self.console.clear()
                    continue
                elif not query:
                    continue
                
                # 询问执行模式
                mode = Prompt.ask(
                    "选择执行模式 [dim](Ctrl+C取消)[/dim]",
                    choices=["智能", "简单", "agent", "simple"],
                    default="智能"
                )
                
                if mode.lower() in ['智能', 'agent']:
                    self.console.print("\n🤖 [bold blue]使用智能Agent模式处理...[/bold blue]")
                    response = await self.process_query_with_agent(query)
                else:
                    self.console.print("\n⚡ [bold yellow]使用简单模式处理...[/bold yellow]")
                    response = await self.process_query(query)
                
                # 显示最终结果
                self.console.print(Panel(
                    response,
                    title="🎯 最终结果",
                    border_style="green"
                ))
                    
            except KeyboardInterrupt:
                consecutive_interrupts += 1
                if consecutive_interrupts >= 2:
                    self.console.print("\n[bold red]检测到连续中断，正在退出程序...[/bold red]")
                    break
                else:
                    self.console.print("\n[yellow]⚠️  操作已取消[/yellow] [dim](再次按 Ctrl+C 退出程序)[/dim]")
                    continue
            except EOFError:
                # 处理 Ctrl+D 或输入流结束
                self.console.print("\n[yellow]👋 检测到输入结束，退出程序[/yellow]")
                break
            except Exception as e:
                consecutive_interrupts = 0  # 重置中断计数
                self.console.print(Panel(
                    f"[red]错误: {str(e)}[/red]",
                    title="❌ 执行错误",
                    border_style="red"
                ))
    
    def _show_help(self):
        """显示帮助信息"""
        help_text = """
[bold blue]MCP智能客户端帮助[/bold blue]

[yellow]基本命令:[/yellow]
• help, h, 帮助     - 显示此帮助信息
• servers, s, 服务器 - 显示服务器状态
• clear, c, 清屏    - 清屏
• quit, q, 退出     - 退出程序

[yellow]执行模式:[/yellow]
• [green]智能模式[/green] - 使用AI Agent进行多步规划和执行
  - 自动分析查询意图
  - 制定详细执行计划
  - 实时显示执行进度
  - 步骤间思考和反思
  - 动态调整策略

• [blue]简单模式[/blue] - 传统的任务分解执行
  - 快速任务分解
  - 顺序执行步骤
  - 基础错误处理

[yellow]示例查询:[/yellow]
• "查询数据库中的用户信息"
• "分析最近的销售数据趋势"
• "生成月度报告"
• "检查系统状态"
"""
        
        self.console.print(Panel(
            help_text,
            title="📖 帮助信息",
            border_style="cyan"
        ))
    
    def _prepare_step_params(self, step: dict, previous_results: List[dict]) -> dict:
        """Prepare parameters for a step, replacing placeholders with previous results"""
        tool_params = step["tool_params"].copy()
        
        if not step.get("depends_on_previous", False) or not previous_results:
            return tool_params
        
        # Get all previous successful results
        successful_results = [r for r in previous_results if r["success"]]
        
        if not successful_results:
            return tool_params
        
        # Replace different types of placeholders
        for param_key, param_value in tool_params.items():
            if isinstance(param_value, str):
                # Replace with latest result
                if "{{previous_result}}" in param_value:
                    latest_result = successful_results[-1]["result"]
                    tool_params[param_key] = param_value.replace("{{previous_result}}", latest_result)
                
                # Replace with specific step result (e.g., {{step_1_result}})
                import re
                step_pattern = r'\{\{step_(\d+)_result\}\}'
                matches = re.findall(step_pattern, param_value)
                for step_num in matches:
                    step_index = int(step_num) - 1
                    if step_index < len(successful_results):
                        step_result = successful_results[step_index]["result"]
                        placeholder = f"{{{{step_{step_num}_result}}}}"
                        tool_params[param_key] = param_value.replace(placeholder, step_result)
                
                # Replace with all previous results combined
                if "{{all_previous_results}}" in param_value:
                    all_results = "\n\n".join([
                        f"步骤 {r['step']['step_number']}: {r['result']}"
                        for r in successful_results
                    ])
                    tool_params[param_key] = param_value.replace("{{all_previous_results}}", all_results)
        
        return tool_params
