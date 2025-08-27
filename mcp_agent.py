import json
import os
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from openai import AsyncOpenAI
from json_repair import repair_json
from datetime import datetime
from base_agent import BaseAgent, ExecutionStep, PlanningResult, StepStatus, AgentState
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, TaskID, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.tree import Tree
from rich.live import Live
import time
from prompt import QUERY_ANALYSIS_PROMPT, PLAN_GENERATION_PROMPT, THINKING_PROMPT, REFLECTION_PROMPT, FINAL_RESPONSE_PROMPT

# 常量定义
DEFAULT_ANALYSIS_RESULT = {
    "intent": "",
    "query_rewrite": "",
    "keywords": [],
    "entities": [],
    "constraints": {
        "time_range": "",
        "language": "",
        "sources": [],
        "file_types": [],
        "limit": 20
    },
    "required_tool_types": [],
    "tool_candidates": [],
    "strategies": ["直接执行"],
    "challenges": [],
    "estimated_steps": 3
}

DEFAULT_PLANNING_RESULT = {
    "steps": [],
    "reasoning": "计划生成失败，使用默认策略",
    "confidence": 0.3,
    "estimated_time_minutes": 5,
    "fallbacks": ["使用备用策略"],
    "stopping_criteria": "获得基本结果"
}

# Use TYPE_CHECKING to avoid circular import
if TYPE_CHECKING:
    from client import MCPClient

class MCPAgent(BaseAgent):
    """基于MCP的智能Agent实现"""
    
    def __init__(self, name: str, config: Dict[str, Any], mcp_client: 'MCPClient'):
        super().__init__(name, config)
        self.mcp_client = mcp_client
        self.openai = AsyncOpenAI(
            api_key=os.getenv("ERNIE_APIKEY"), 
            base_url=os.getenv("ERNIE_BASE_URL")
        )
        self.console = Console()
        self.progress = None
        self.current_task_id = None
    
    async def _call_openai(self, prompt: str, model_type: str = "thinking", temperature: float = 0.3, max_tokens: int = 4096) -> str:
        """通用的OpenAI调用方法"""
        model_map = {
            "thinking": os.getenv("ERNIE_THINKING_MODEL"),
            "chat": os.getenv("ERNIE_CHAT_MODEL")
        }
        
        model = model_map.get(model_type, os.getenv("ERNIE_THINKING_MODEL"))
        
        try:
            completion = await self.openai.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return completion.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI调用失败: {str(e)}")
    
    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """处理用户查询的主要入口点，带进度显示"""
        try:
            self.state = AgentState.PLANNING
            
            # 显示开始信息
            self.console.print(Panel.fit(
                f"🤖 [bold blue]智能Agent开始处理查询[/bold blue]\n\n查询: {query}",
                title="Agent启动",
                border_style="blue"
            ))
            
            # 1. 思考和规划阶段
            self.console.print("\n[bold magenta]📋 阶段1: 智能规划[/bold magenta]")
            planning_result = await self._plan_execution_with_display(query, context)
            if not planning_result:
                return "无法为此查询制定执行计划"
            
            self.current_plan = planning_result
            self.current_step_index = 0
            
            # 显示规划结果
            self._display_plan(planning_result)
            
            # 2. 执行阶段
            self.state = AgentState.EXECUTING
            self.console.print("\n[bold green]⚡ 阶段2: 执行计划[/bold green]")
            execution_results = await self._execute_plan_with_progress()
            
            # 3. 最终反思和总结
            self.state = AgentState.REFLECTING
            self.console.print("\n[bold yellow]🤔 阶段3: 反思总结[/bold yellow]")
            final_response = await self._generate_final_response(query, execution_results)
            
            self.state = AgentState.COMPLETED
            
            # 显示完成信息
            self.console.print(Panel.fit(
                f"✅ [bold green]任务执行完成[/bold green]\n\n共执行 {len(execution_results)} 个步骤",
                title="执行完成",
                border_style="green"
            ))
            
            return final_response
            
        except Exception as e:
            self.state = AgentState.ERROR
            self.console.print(Panel.fit(
                f"❌ [bold red]执行过程中发生错误[/bold red]\n\n错误: {str(e)}",
                title="执行失败",
                border_style="red"
            ))
            return f"执行过程中发生错误: {str(e)}"
    
    async def _plan_execution_with_display(self, query: str, context: Optional[Dict[str, Any]] = None) -> Optional[PlanningResult]:
        """带显示的规划执行步骤"""
        with self.console.status("[bold blue]正在分析查询和收集工具信息...") as status:
            # 收集可用工具信息
            available_tools = await self._get_available_tools()
            status.update("[bold blue]正在深度分析用户意图...")
            
            # 分析查询和上下文
            analysis = await self._analyze_query(query, context, available_tools)
            status.update("[bold blue]正在制定最优执行计划...")
            
            # 生成执行计划
            planning_result = await self._generate_plan(query, analysis, available_tools)
        
        # 记录规划过程
        self.memory.conversation_history.append({
            "type": "planning",
            "query": query,
            "context": context,
            "planning_result": planning_result,
            "timestamp": datetime.now()
        })
        
        return planning_result
    
    def _display_plan(self, planning_result: PlanningResult):
        """显示执行计划"""
        # 创建计划树状图
        tree = Tree("📋 [bold blue]执行计划[/bold blue]")
        
        # 添加规划推理
        reasoning_branch = tree.add("🧠 [yellow]规划推理[/yellow]")
        reasoning_branch.add(planning_result.reasoning)
        
        # 添加置信度和预估时间
        confidence_branch = tree.add(f"📊 [cyan]置信度: {planning_result.confidence:.1%}[/cyan]")
        if planning_result.estimated_time_minutes:
            time_branch = tree.add(f"⏱️ [cyan]预估时间: {planning_result.estimated_time_minutes}分钟[/cyan]")
        
        # 添加执行步骤
        steps_branch = tree.add("📝 [green]执行步骤[/green]")
        for i, step in enumerate(planning_result.steps, 1):
            # 确保 description 是字符串类型
            if isinstance(step.description, dict):
                # 如果是字典，转换为可读的字符串
                step_description = f"{step.description.get('description', '未知步骤')}"
            elif isinstance(step.description, str):
                step_description = step.description
            else:
                # 其他类型转换为字符串
                step_description = str(step.description)
            
            step_text = f"步骤{i}: {step_description}"
            if step.step_type:
                step_text += f" [dim]({step.step_type})[/dim]"
            if step.tool_name:
                step_text += f" [dim](工具: {step.tool_name})[/dim]"
            if step.server_name:
                step_text += f" [dim](服务器: {step.server_name})[/dim]"
            steps_branch.add(step_text)
        
        # 添加回退策略
        if planning_result.fallbacks:
            fallback_branch = tree.add("🔄 [orange3]回退策略[/orange3]")
            for fallback in planning_result.fallbacks:
                fallback_branch.add(fallback)
        
        # 添加停止条件
        if planning_result.stopping_criteria:
            criteria_branch = tree.add("🎯 [magenta]停止条件[/magenta]")
            criteria_branch.add(planning_result.stopping_criteria)
        
        self.console.print(tree)
        self.console.print()
    
    async def _execute_plan_with_progress(self) -> List[ExecutionStep]:
        """带进度条的执行计划"""
        execution_results = []
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=self.console
        ) as progress:
            
            # 创建总体进度任务
            total_task = progress.add_task(
                "[cyan]总体进度", 
                total=len(self.current_plan.steps)
            )
            
            while self.current_step_index < len(self.current_plan.steps):
                step = self.current_plan.steps[self.current_step_index]
                
                # 创建当前步骤的进度任务
                step_task = progress.add_task(
                    f"[green]步骤{self.current_step_index + 1}: {step.description[:50]}...",
                    total=100
                )
                
                # 执行前思考
                progress.update(step_task, advance=20, description=f"[yellow]思考中: {step.description[:40]}...")
                await self._think_before_execution(step)
                
                # 执行步骤
                progress.update(step_task, advance=30, description=f"[blue]执行中: {step.description[:40]}...")
                step.status = StepStatus.EXECUTING
                start_time = time.time()
                
                try:
                    result = await self._execute_step(step)
                    step.result = result
                    # 确保状态更新
                    if result and result.strip():
                        step.status = StepStatus.COMPLETED
                    else:
                        step.status = StepStatus.FAILED
                        step.error = "步骤执行完成但未返回有效结果"
                    step.execution_time = time.time() - start_time
                    progress.update(step_task, advance=30, description=f"[green]✅ 完成: {step.description[:40]}...")
                    
                except Exception as e:
                    step.error = str(e)
                    step.status = StepStatus.FAILED
                    step.execution_time = time.time() - start_time
                    progress.update(step_task, advance=30, description=f"[red]❌ 失败: {step.description[:40]}...")
                
                # 执行后反思
                progress.update(step_task, advance=20, description=f"[magenta]反思中: {step.description[:40]}...")
                await self._reflect_on_step(step)
                
                # 完成当前步骤
                progress.update(step_task, completed=100)
                progress.remove_task(step_task)
                
                execution_results.append(step)
                self.memory.execution_history.append(step)
                
                # 更新总体进度
                progress.update(total_task, advance=1)
                
                # 显示步骤结果摘要
                self._display_step_result(step, self.current_step_index + 1)
                
                # 根据反思结果决定是否需要重新规划
                if await self._should_replan(step, execution_results):
                    self.console.print("[yellow]⚠️  检测到需要重新规划，正在调整策略...[/yellow]")
                    new_plan = await self._replan(execution_results)
                    if new_plan:
                        self.current_plan = new_plan
                        self.current_step_index = 0
                        # 重新设置进度条
                        progress.update(total_task, completed=0, total=len(new_plan.steps))
                        continue
                
                self.current_step_index += 1
        
        return execution_results
    
    def _display_step_result(self, step: ExecutionStep, step_number: int):
        """显示步骤执行结果"""
        if step.status == StepStatus.COMPLETED:
            status_icon = "✅"
            status_color = "green"
            status_text = "成功"
        elif step.status == StepStatus.FAILED:
            status_icon = "❌"
            status_color = "red"
            status_text = "失败"
        else:
            status_icon = "⏳"
            status_color = "yellow"
            status_text = "进行中"
        
        # 创建结果表格
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("属性", style="bold")
        table.add_column("值")
        
        table.add_row("状态", f"[{status_color}]{status_icon} {status_text}[/{status_color}]")
        table.add_row("执行时间", f"{step.execution_time:.2f}秒" if step.execution_time else "未知")
        
        if step.result:
            result_preview = step.result[:100] + "..." if len(step.result) > 100 else step.result
            table.add_row("结果预览", result_preview)
        
        if step.error:
            table.add_row("错误信息", f"[red]{step.error}[/red]")
        
        if step.confidence_score:
            confidence_color = "green" if step.confidence_score > 0.7 else "yellow" if step.confidence_score > 0.4 else "red"
            table.add_row("置信度", f"[{confidence_color}]{step.confidence_score:.1%}[/{confidence_color}]")
        
        panel = Panel(
            table,
            title=f"步骤 {step_number} 结果",
            border_style=status_color
        )
        
        self.console.print(panel)
        self.console.print()
    
    async def _get_available_tools(self) -> List[Dict[str, Any]]:
        """获取所有MCP服务器的可用工具"""
        all_tools = []
        
        for server_name, session in self.mcp_client.sessions.items():
            try:
                response = await session.list_tools()
                server_tools = [{
                    "server": server_name,
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema
                } for tool in response.tools]
                all_tools.extend(server_tools)
            except Exception as e:
                print(f"获取服务器 {server_name} 工具失败: {str(e)}")
        
        return all_tools
    
    async def _analyze_query(self, query: str, context: Optional[Dict[str, Any]], available_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析用户查询"""
        analysis_prompt = QUERY_ANALYSIS_PROMPT.format(
            query=query,
            context=json.dumps(context or {}, ensure_ascii=False, indent=2),
            available_tools=json.dumps([{"name": t["name"], "description": t["description"]} for t in available_tools], ensure_ascii=False, indent=2)
        )
        
        try:
            response_text = await self._call_openai(analysis_prompt, "thinking", 0.3)
            response_text = repair_json(response_text)
            return json.loads(response_text)
            
        except Exception as e:
            print(f"查询分析失败: {str(e)}")
            result = DEFAULT_ANALYSIS_RESULT.copy()
            result["intent"] = query
            return result
    
    async def _generate_plan(self, query: str, analysis: Dict[str, Any], available_tools: List[Dict[str, Any]]) -> PlanningResult:
        """生成执行计划"""
        planning_prompt = PLAN_GENERATION_PROMPT.format(
            query=query,
            analysis=json.dumps(analysis, ensure_ascii=False, indent=2),
            available_tools=json.dumps(available_tools, ensure_ascii=False, indent=2)
        )
        
        try:
            response_text = await self._call_openai(planning_prompt, "thinking", 0.4)
            response_text = repair_json(response_text)
            plan_data = json.loads(response_text)
            
            # 转换为ExecutionStep对象
            steps = []
            for step_data in plan_data.get("steps", []):
                # 确保 description 是字符串类型
                description = step_data.get("description", "")
                if isinstance(description, dict):
                    description = description.get("description", str(description))
                elif not isinstance(description, str):
                    description = str(description)
                
                step = ExecutionStep(
                    step_id=step_data["step_id"],
                    description=description,  # 使用处理后的 description
                    tool_name=step_data.get("tool_name"),
                    server_name=step_data.get("server_name"),
                    parameters=step_data.get("parameters", {}),
                    step_type=step_data.get("type")  # 新增字段
                )
                steps.append(step)
            
            return PlanningResult(
                steps=steps,
                reasoning=plan_data.get("reasoning", ""),
                confidence=plan_data.get("confidence", 0.5),
                estimated_time=plan_data.get("estimated_time"),
                estimated_time_minutes=plan_data.get("estimated_time_minutes"),  # 新增
                fallbacks=plan_data.get("fallbacks", []),  # 新增
                stopping_criteria=plan_data.get("stopping_criteria")  # 新增
            )
            
        except Exception as e:
            print(f"计划生成失败: {str(e)}")
            # 返回默认计划
            return PlanningResult(
                steps=DEFAULT_PLANNING_RESULT["steps"],
                reasoning=DEFAULT_PLANNING_RESULT["reasoning"],
                confidence=DEFAULT_PLANNING_RESULT["confidence"],
                estimated_time_minutes=DEFAULT_PLANNING_RESULT["estimated_time_minutes"],
                fallbacks=DEFAULT_PLANNING_RESULT["fallbacks"],
                stopping_criteria=DEFAULT_PLANNING_RESULT["stopping_criteria"]
            )
    
    async def _execute_step(self, step: ExecutionStep) -> str:
        """执行单个步骤"""
        if not step.tool_name or not step.server_name:
            raise Exception("步骤缺少必要的工具或服务器信息")
        
        if step.server_name not in self.mcp_client.sessions:
            raise Exception(f"服务器 {step.server_name} 不可用")
        
        session = self.mcp_client.sessions[step.server_name]
        
        # 准备参数（处理占位符）
        prepared_params = self._prepare_step_params(step)
        
        # 执行工具调用
        result = await session.call_tool(step.tool_name, prepared_params)
        
        if result.content and len(result.content) > 0:
            return result.content[0].text
        else:
            return "执行完成，但没有返回内容"
    
    def _prepare_step_params(self, step: ExecutionStep) -> Dict[str, Any]:
        """准备步骤参数，处理占位符"""
        params = step.parameters.copy()
        
        # 获取之前的成功结果
        previous_results = [s for s in self.memory.execution_history if s.status == StepStatus.COMPLETED]
        
        if not previous_results:
            return params
        
        # 替换占位符
        for key, value in params.items():
            if isinstance(value, str):
                if "{{previous_result}}" in value:
                    latest_result = previous_results[-1].result or ""
                    params[key] = value.replace("{{previous_result}}", latest_result)
                
                # 处理特定步骤结果引用
                import re
                step_pattern = r'\{\{step_([^}]+)_result\}\}'
                matches = re.findall(step_pattern, value)
                for step_id in matches:
                    step_result = next((s.result for s in previous_results if s.step_id == step_id), "")
                    placeholder = f"{{{{step_{step_id}_result}}}}"
                    params[key] = value.replace(placeholder, step_result or "")
        
        return params
    
    async def _generate_thinking(self, context: Dict[str, Any]) -> str:
        """生成思考内容"""
        thinking_prompt = THINKING_PROMPT.format(
            context=json.dumps(context, ensure_ascii=False, indent=2, default=str)
        )
        
        try:
            return await self._call_openai(thinking_prompt, "thinking", 0.6)
            
        except Exception as e:
            return f"思考过程出错: {str(e)}"
    
    async def _generate_reflection(self, context: Dict[str, Any]) -> str:
        """生成反思内容"""
        reflection_prompt = REFLECTION_PROMPT.format(
            context=json.dumps(context, ensure_ascii=False, indent=2, default=str)
        )
        
        try:
            return await self._call_openai(reflection_prompt, "thinking", 0.6)
            
        except Exception as e:
            return f"反思过程出错: {str(e)}"
    
    async def _generate_final_response(self, query: str, execution_results: List[ExecutionStep]) -> str:
        """生成最终响应"""
        # 提取成功执行的结果内容 - 修复状态检查
        successful_results = []
        for step in execution_results:
            # 修改条件：包含有结果的步骤，不仅仅是COMPLETED状态
            if step.result and step.result.strip():
                # 额外检查：排除明显的错误结果
                if not step.result.startswith("抱歉") and not step.result.startswith("无法"):
                    successful_results.append(step.result)
            # 如果没有成功结果，但步骤状态是COMPLETED，也尝试包含
            elif step.status == StepStatus.COMPLETED and step.result:
                successful_results.append(step.result)
        
        # 合并所有成功的结果
        combined_results = "\n\n".join(successful_results)
        
        final_prompt = FINAL_RESPONSE_PROMPT.format(
            query=query,
            combined_results=combined_results
        )
        
        try:
            return await self._call_openai(final_prompt, "chat", 0.6)
            
        except Exception as e:
            # 备用方案：直接返回合并的结果
            if combined_results:
                return combined_results
            else:
                return f"抱歉，无法获取到关于'{query}'的相关信息。"