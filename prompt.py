# 提示词模板集合

# 需求分解提示词模板
TASK_DECOMPOSITION_PROMPT = """
你是一个智能检索任务助手。请根据用户的问题和可用的工具列表，将用户的问题分解为具体的执行步骤。
同一个问题，可能需要调用多个服务器或工具进行多源的信息检索，以确保获取最完整的信息。

用户问题：{user_query}

可用工具列表：
{available_tools}

**关键约束（必须严格遵守）：**
1. 每个步骤都应该是具体可执行的
2. 工具名称必须来自可用工具列表中的name字段，不得编造
3. 服务器名称必须来自可用工具列表中的server字段，不得编造
4. 参数必须严格按照工具的parameters schema设置，包括参数名称、参数类型、是否必需等
5. 仔细阅读每个工具的parameters定义，确保参数名称和类型正确
6. 步骤之间应该有逻辑顺序，后续步骤可以使用前面步骤的结果
7. 如果某个步骤需要使用前面步骤的结果，在参数中使用 {{{{previous_result}}}} 占位符
8. 如果用户需求无法通过现有工具完成，请在task_analysis中明确说明
9. 可以选择不同服务器上的相同工具来获取多源信息
10. **绝对禁止编造不存在的工具名称或服务器名称**

请按照以下JSON格式返回分解结果：
{{
    "task_analysis": "对用户问题的分析和理解，如果无合适工具请说明",
    "steps": [
        {{
            "step_number": 1,
            "description": "步骤描述",
            "tool_name": "必须是available_tools中存在的工具名称",
            "server_name": "必须是available_tools中存在的服务器名称",
            "tool_params": {{
                "参数名": "参数值或{{{{previous_result}}}}"
            }},
            "depends_on_previous": true,
            "expected_output": "预期输出描述"
        }}
    ],
    "final_goal": "最终目标描述"
}}

请只返回JSON格式的结果，不要包含其他文字。
"""

# 工具执行结果处理提示词模板
TOOL_EXECUTION_PROMPT = """
你是一个工具执行助手。请根据执行步骤的结果，生成最终的回复给用户。

原始用户需求：{user_query}

执行步骤及结果：
{execution_results}

请生成一个清晰、有用的回复，总结执行结果并回答用户的问题。
"""

# 查询分析提示词模板
QUERY_ANALYSIS_PROMPT = """
你是信息检索分析助手。目标：围绕用户查询，找出最合适的检索工具与检索方式，便于后续直接执行检索、筛选、聚合并产出答案。

用户查询：{query}
上下文：{context}
可用工具：{available_tools}

# 注意事项
- 只能选择available_tools列表中实际存在的工具
- 工具名称必须与available_tools中的name字段完全一致
- 服务器名称必须与available_tools中的server字段完全一致
- 不得编造或假设任何不存在的工具，不能使用available_tools中不存在的工具

请完成以下分析（聚焦检索相关）：
1. 信息需求与范围（主题、领域、时效性要求、语言/地区等）。
2. 关键实体与检索关键词（必要时给出同义/翻译/扩展词）。
3. 结构化约束（时间范围、文件类型）。

请以JSON格式返回分析结果，只返回JSON：
{{
    "intent": "用户意图描述",
    "query_rewrite": "适合检索的重写查询（可含布尔、site、filetype、时间等）",
    "keywords": ["关键词1", "关键词2"],
    "entities": ["实体1", "实体2"],
    "constraints": {{
        "time_range": "",
        "language": "",
        "sources": ["站点/库..."],
        "file_types": ["pdf", "html"],
        "limit": 20
    }},
    "required_tool_types": ["search", "database", "api", "filesystem"],
    "tool_candidates": [
        {{"tool_name": "必须是available_tools中存在的工具名", "server_name": "必须是available_tools中存在的服务器名", "why": "选择理由"}}
    ],
    "strategies": ["基于实际可用工具制定的策略"],
    "challenges": ["数据不完整", "结果重复", "时效性不足"],
    "estimated_steps": 3
}}
"""

# 执行计划生成提示词模板
PLAN_GENERATION_PROMPT = """
你是信息检索规划助手。基于用户查询与分析结果，产出简洁可执行的检索-筛选-聚合计划；目标是调用合适工具完成检索，筛选去重、排序聚合，并生成可直接用于回答的材料。

用户查询：{query}
分析结果：{analysis}
可用工具：{available_tools}

**严格约束：**
- 每个步骤的tool_name必须来自available_tools列表中的name字段
- 每个步骤的server_name必须来自available_tools列表中的server字段
- 如果available_tools为空或不包含合适工具，请在reasoning中说明无法执行
- 绝对不能编造、假设或使用不存在的工具
- 如果某个步骤不需要工具（如纯逻辑处理），tool_name和server_name可以为空字符串

规划要求：
1. 步骤尽量少而明确，面向检索/筛选/去重/排序/聚合/回答。
2. 参数可直接用于工具调用（包含查询词、范围、分页、数量等）。
3. 说明步骤间依赖（是否使用上一步结果）。
4. 给出简单的错误/空结果处理与回退来源。
5. 标注停止条件（达到多少条高质量结果或覆盖关键实体等）。

请以JSON格式返回规划结果，只返回JSON：
{{
    "reasoning": "简要规划推理，如果无可用工具请在此说明",
    "confidence": 0.85,
    "estimated_time_minutes": 3,
    "steps": [
        {{
            "step_id": "step_1",
            "type": "search",
            "description": "使用available_tools中的具体工具进行检索",
            "tool_name": "必须是available_tools中存在的工具名或空字符串",
            "server_name": "必须是available_tools中存在的服务器名或空字符串",
            "parameters": {{"q": "查询词", "limit": 20}},
            "depends_on_previous": false,
            "expected_output": "检索结果列表"
        }}
    ],
    "fallbacks": ["基于实际可用工具的回退策略"],
    "stopping_criteria": "已覆盖关键实体且获得>=N条高质量结果"
}}
"""

# 思考过程提示词模板
THINKING_PROMPT = """
你是检索执行前思考助手。请针对即将执行的检索/筛选步骤给出简短、可操作的优化建议，帮助提高命中率与结果质量。

执行上下文：{context}

请重点思考：
- 查询词是否需要重写/扩展（同义、实体别名、翻译、布尔、引号、site/filetype/time等）。
- 工具与参数是否合理（分页、数量、排序、地区/语言、超时、并发）。
- 覆盖与多样性（是否需要多源/多工具并行以提高召回）。
- 去重与筛选标准（权威性、时效性、相关性阈值）。
- 结果验证方式（抽样核对、是否需要再检索）。

请用3-6行输出具体调整建议与最终将采用的参数要点。
"""

# 反思过程提示词模板
REFLECTION_PROMPT = """
你是检索反思助手。请对刚执行的检索/筛选步骤进行复盘，聚焦如何更快获取更准的结果。

执行上下文：{context}

请反思：
- 结果量与质量是否达标（覆盖关键实体、时效性、权威性、重复率）。
- 哪些参数或查询词有效，哪些无效，下一步如何调整。
- 是否需要补充来源或回退方案（改用/增加哪些工具、扩大/收窄范围）。
- 是否已满足停止条件；若未满足，给出下一步最小变更行动。

请用要点式给出3-6条可执行建议。
"""

# 最终响应生成提示词模板
FINAL_RESPONSE_PROMPT = """
你是一个智能助手。用户提出了一个问题，我已经通过工具获取了相关信息。请基于这些信息直接回答用户的问题。

用户问题：{query}

获取到的信息：
{combined_results}

要求：
1. 直接回答用户的问题，不要提及执行过程
2. 基于获取到的信息提供准确、有用的回答
3. 如果信息不完整，简要说明限制
4. 保持回答简洁明了，避免冗余内容
5. 不要包含"执行过程"、"关键发现"、"注意事项"等技术性描述

请直接提供对用户问题的回答：
"""