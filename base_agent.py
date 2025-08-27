from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from datetime import datetime

class StepStatus(Enum):
    PENDING = "pending"
    THINKING = "thinking"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    REFLECTING = "reflecting"

class AgentState(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class ExecutionStep:
    """表示一个执行步骤"""
    step_id: str
    description: str
    tool_name: Optional[str] = None
    server_name: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: StepStatus = StepStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    thinking: Optional[str] = None
    reflection: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    execution_time: Optional[float] = None
    confidence_score: Optional[float] = None
    # 新增字段
    step_type: Optional[str] = None  # search, filter, aggregate等

@dataclass
class PlanningResult:
    """规划结果"""
    steps: List[ExecutionStep]
    reasoning: str
    confidence: float
    estimated_time: Optional[float] = None
    alternative_plans: List[List[ExecutionStep]] = field(default_factory=list)
    # 新增字段
    estimated_time_minutes: Optional[float] = None
    fallbacks: List[str] = field(default_factory=list)
    stopping_criteria: Optional[str] = None

@dataclass
class AgentMemory:
    """Agent的记忆系统"""
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    execution_history: List[ExecutionStep] = field(default_factory=list)
    learned_patterns: Dict[str, Any] = field(default_factory=dict)
    context_summary: Optional[str] = None
    last_reflection: Optional[str] = None

class BaseAgent(ABC):
    """基础Agent类，实现多步规划、思考、执行、反思的架构"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.state = AgentState.IDLE
        self.memory = AgentMemory()
        self.current_plan: Optional[PlanningResult] = None
        self.current_step_index = 0
        self.max_reflection_cycles = config.get("max_reflection_cycles", 3)
        self.enable_dynamic_replanning = config.get("enable_dynamic_replanning", True)
        
    async def process_query(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """处理用户查询的主要入口点"""
        try:
            self.state = AgentState.PLANNING
            
            # 1. 思考和规划阶段
            planning_result = await self._plan_execution(query, context)
            if not planning_result:
                return "无法为此查询制定执行计划"
            
            self.current_plan = planning_result
            self.current_step_index = 0
            
            # 2. 执行阶段
            self.state = AgentState.EXECUTING
            execution_results = await self._execute_plan()
            
            # 3. 最终反思和总结
            self.state = AgentState.REFLECTING
            final_response = await self._generate_final_response(query, execution_results)
            
            self.state = AgentState.COMPLETED
            return final_response
            
        except Exception as e:
            self.state = AgentState.ERROR
            return f"执行过程中发生错误: {str(e)}"
    
    async def _plan_execution(self, query: str, context: Optional[Dict[str, Any]] = None) -> Optional[PlanningResult]:
        """规划执行步骤"""
        # 收集可用工具信息
        available_tools = await self._get_available_tools()
        
        # 分析查询和上下文
        analysis = await self._analyze_query(query, context, available_tools)
        
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
    
    async def _execute_plan(self) -> List[ExecutionStep]:
        """执行计划中的步骤"""
        execution_results = []
        
        while self.current_step_index < len(self.current_plan.steps):
            step = self.current_plan.steps[self.current_step_index]
            
            # 执行前思考
            await self._think_before_execution(step)
            
            # 执行步骤
            step.status = StepStatus.EXECUTING
            start_time = time.time()
            
            try:
                result = await self._execute_step(step)
                step.result = result
                step.status = StepStatus.COMPLETED
                step.execution_time = time.time() - start_time
                
            except Exception as e:
                step.error = str(e)
                step.status = StepStatus.FAILED
                step.execution_time = time.time() - start_time
            
            # 执行后反思
            await self._reflect_on_step(step)
            
            execution_results.append(step)
            self.memory.execution_history.append(step)
            
            # 根据反思结果决定是否需要重新规划
            if await self._should_replan(step, execution_results):
                new_plan = await self._replan(execution_results)
                if new_plan:
                    self.current_plan = new_plan
                    self.current_step_index = 0
                    continue
            
            self.current_step_index += 1
        
        return execution_results
    
    async def _think_before_execution(self, step: ExecutionStep):
        """执行前的思考过程"""
        step.status = StepStatus.THINKING
        
        # 分析当前情况
        context = {
            "previous_steps": self.memory.execution_history[-3:],  # 最近3步
            "current_step": step,
            "remaining_steps": self.current_plan.steps[self.current_step_index + 1:]
        }
        
        # 生成思考内容
        thinking_result = await self._generate_thinking(context)
        step.thinking = thinking_result
        
        # 根据思考结果调整参数
        if thinking_result:
            adjusted_params = await self._adjust_parameters_based_on_thinking(step, thinking_result)
            if adjusted_params:
                step.parameters.update(adjusted_params)
    
    async def _reflect_on_step(self, step: ExecutionStep):
        """对执行步骤进行反思"""
        step.status = StepStatus.REFLECTING
        
        # 评估执行结果
        reflection_context = {
            "step": step,
            "previous_steps": self.memory.execution_history[-5:],
            "original_plan": self.current_plan
        }
        
        reflection = await self._generate_reflection(reflection_context)
        step.reflection = reflection
        
        # 计算置信度分数
        step.confidence_score = await self._calculate_confidence_score(step)
        
        # 更新学习模式
        await self._update_learned_patterns(step)
    
    async def _should_replan(self, current_step: ExecutionStep, execution_results: List[ExecutionStep]) -> bool:
        """判断是否需要重新规划"""
        if not self.enable_dynamic_replanning:
            return False
        
        # 如果当前步骤失败且是关键步骤
        if current_step.status == StepStatus.FAILED:
            return True
        
        # 如果置信度过低
        if current_step.confidence_score and current_step.confidence_score < 0.3:
            return True
        
        # 如果反思建议重新规划
        if current_step.reflection and "重新规划" in current_step.reflection:
            return True
        
        return False
    
    # 抽象方法，需要子类实现
    @abstractmethod
    async def _get_available_tools(self) -> List[Dict[str, Any]]:
        """获取可用工具列表"""
        pass
    
    @abstractmethod
    async def _analyze_query(self, query: str, context: Optional[Dict[str, Any]], available_tools: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析查询"""
        pass
    
    @abstractmethod
    async def _generate_plan(self, query: str, analysis: Dict[str, Any], available_tools: List[Dict[str, Any]]) -> PlanningResult:
        """生成执行计划"""
        pass
    
    @abstractmethod
    async def _execute_step(self, step: ExecutionStep) -> str:
        """执行单个步骤"""
        pass
    
    @abstractmethod
    async def _generate_thinking(self, context: Dict[str, Any]) -> str:
        """生成思考内容"""
        pass
    
    @abstractmethod
    async def _generate_reflection(self, context: Dict[str, Any]) -> str:
        """生成反思内容"""
        pass
    
    @abstractmethod
    async def _generate_final_response(self, query: str, execution_results: List[ExecutionStep]) -> str:
        """生成最终响应"""
        pass
    
    async def _adjust_parameters_based_on_thinking(self, step: ExecutionStep, thinking: str) -> Optional[Dict[str, Any]]:
        """根据思考结果调整参数"""
        # 默认实现，子类可以重写
        return None
    
    async def _calculate_confidence_score(self, step: ExecutionStep) -> float:
        """计算置信度分数"""
        # 默认实现，基于执行状态
        if step.status == StepStatus.COMPLETED and not step.error:
            return 0.8
        elif step.status == StepStatus.FAILED:
            return 0.1
        else:
            return 0.5
    
    async def _update_learned_patterns(self, step: ExecutionStep):
        """更新学习模式"""
        # 记录成功/失败的模式
        pattern_key = f"{step.tool_name}_{step.status.value}"
        if pattern_key not in self.memory.learned_patterns:
            self.memory.learned_patterns[pattern_key] = []
        
        self.memory.learned_patterns[pattern_key].append({
            "step": step,
            "timestamp": datetime.now()
        })
    
    async def _replan(self, execution_results: List[ExecutionStep]) -> Optional[PlanningResult]:
        """重新规划"""
        # 分析失败原因
        failure_analysis = await self._analyze_failures(execution_results)
        
        # 生成新计划
        # 这里需要根据具体实现来调整
        return None
    
    async def _analyze_failures(self, execution_results: List[ExecutionStep]) -> Dict[str, Any]:
        """分析失败原因"""
        failures = [step for step in execution_results if step.status == StepStatus.FAILED]
        return {
            "failure_count": len(failures),
            "failure_steps": failures,
            "common_errors": [step.error for step in failures if step.error]
        }