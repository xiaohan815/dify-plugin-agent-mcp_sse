"""
ReAct (Reasoning and Acting) Agent 策略实现
这个模块实现了一个基于 ReAct 模式的智能代理策略,它能够:
1. 通过思考(Reasoning)来理解问题
2. 采取行动(Acting)来执行工具调用
3. 观察(Observation)执行结果
4. 最终给出答案
"""

import json
import time
from collections.abc import Generator, Mapping
from typing import Any, Optional, cast

import pydantic
from dify_plugin.entities.agent import AgentInvokeMessage
from dify_plugin.entities.model.llm import LLMModelConfig, LLMUsage
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    SystemPromptMessage,
    UserPromptMessage,
    PromptMessageTool,
)
from dify_plugin.entities.tool import (
    LogMetadata,
    ToolInvokeMessage,
    ToolParameter,
    ToolProviderType,
)
from dify_plugin.interfaces.agent import (
    AgentModelConfig,
    AgentScratchpadUnit,
    AgentStrategy,
    ToolEntity,
)
from pydantic import BaseModel

from output_parser.cot_output_parser import CotAgentOutputParser
from prompt.template import REACT_PROMPT_TEMPLATES
from utils.mcp_client import McpClients

# 忽略观察结果的提供商列表
ignore_observation_providers = ["wenxin"]


class ReActParams(BaseModel):
    """
    ReAct 策略的参数配置类
    包含:
    - query: 用户查询
    - instruction: 系统指令
    - model: 模型配置
    - tools: 可用工具列表
    - mcp_servers_config: MCP服务器配置
    - maximum_iterations: 最大迭代次数
    """
    query: str
    instruction: str
    model: AgentModelConfig
    tools: list[ToolEntity] | None
    mcp_servers_config: str | None
    maximum_iterations: int = 3


class AgentPromptEntity(BaseModel):
    """
    Agent 提示词实体类
    包含:
    - first_prompt: 首次提示词
    - next_iteration: 后续迭代的提示词
    """
    first_prompt: str
    next_iteration: str


class ReActAgentStrategy(AgentStrategy):
    """
    ReAct Agent 策略实现类
    实现了基于 ReAct 模式的智能代理策略
    """
    def __init__(self, runtime, session):
        """
        初始化 ReAct Agent 策略
        Args:
            runtime: 运行时环境
            session: 会话对象
        """
        super().__init__(runtime, session)
        self.query = ""
        self.instruction = ""
        self.history_prompt_messages = []
        self.prompt_messages_tools = []

    @property
    def _user_prompt_message(self) -> UserPromptMessage:
        """获取用户提示消息"""
        return UserPromptMessage(content=self.query)

    @property
    def _system_prompt_message(self) -> SystemPromptMessage:
        """
        获取系统提示消息
        包含:
        1. 基础提示词
        2. 可用工具信息
        3. 工具名称列表
        """
        prompt_entity = AgentPromptEntity(
            first_prompt=REACT_PROMPT_TEMPLATES["english"]["chat"]["prompt"],
            next_iteration=REACT_PROMPT_TEMPLATES["english"]["chat"][
                "agent_scratchpad"
            ],
        )
        if not prompt_entity:
            raise ValueError("Agent prompt configuration is not set")
        first_prompt = prompt_entity.first_prompt

        system_prompt = (
            first_prompt.replace("{{instruction}}", self.instruction)
            .replace(
                "{{tools}}",
                json.dumps(
                    [
                        tool.model_dump(mode="json")
                        for tool in self._prompt_messages_tools
                    ],
                    ensure_ascii=False
                ),
            )
            .replace(
                "{{tool_names}}",
                ", ".join([tool.name for tool in self._prompt_messages_tools]),
            )
        )
        # print(f"system_prompt: {system_prompt}")
        return SystemPromptMessage(content=system_prompt)

    def _invoke(self, parameters: dict[str, Any]) -> Generator[AgentInvokeMessage]:
        """
        执行 ReAct agent 应用
        主要流程:
        1. 初始化参数和状态
        2. 循环执行思考-行动-观察过程
        3. 直到达到最大迭代次数或得到最终答案
        
        Args:
            parameters: 参数字典,包含查询、指令、模型配置等
            
        Yields:
            AgentInvokeMessage: 执行过程中的消息
        """

        try:
            react_params = ReActParams(**parameters)
        except pydantic.ValidationError as e:
            raise ValueError(f"Invalid parameters: {e!s}") from e

        # Init parameters
        self.query = react_params.query
        self.instruction = react_params.instruction
        print(f"用户提问: {self.query}")
        # print(self.instruction)
        # print(react_params.model)
        # print(react_params.tools)
        # print(react_params.mcp_servers_config)
        print(f"最大迭代次数: {react_params.maximum_iterations}")
        # agent_scratchpad 用于存储代理的思考和行动历史记录
        agent_scratchpad = []
        iteration_step = 1
        max_iteration_steps = react_params.maximum_iterations
        run_agent_state = True
        llm_usage: dict[str, Optional[LLMUsage]] = {"usage": None}
        final_answer = ""
        # 初始化提示消息列表，用于构建完整的对话历史
        prompt_messages = []

        # Init model
        model = react_params.model
        # 初始化模型停止词列表
        # 从模型配置中获取停止词，用于控制 LLM 在遇到这些词时停止生成
        # 如果模型配置中没有指定停止词，则使用空列表
        stop = (
            react_params.model.completion_params.get("stop", [])
            if react_params.model.completion_params
            else []
        )
        if (
                "Observation" not in stop
                and model.provider not in ignore_observation_providers
        ):
            stop.append("Observation")

        # Init prompts
        self.history_prompt_messages = model.history_prompt_messages

        # convert tools into ModelRuntime Tool format
        tools = react_params.tools
        tool_instances = {tool.identity.name: tool for tool in tools} if tools else {}

        # Fetch MCP tools
        mcp_clients = None
        mcp_tools = []
        mcp_tool_instances = {}
        servers_config_json = react_params.mcp_servers_config
        if servers_config_json:
            try:
                servers_config = json.loads(servers_config_json)
            except json.JSONDecodeError as e:
                raise ValueError(f"mcp_servers_config must be a valid JSON string: {e}")
            mcp_clients = McpClients(servers_config)
            mcp_tools = mcp_clients.fetch_tools()
            mcp_tool_instances = {tool.get("name"): tool for tool in mcp_tools} if mcp_tools else {}

        react_params.model.completion_params = (
                react_params.model.completion_params or {}
        )
        # convert tools into ModelRuntime Tool format
        prompt_messages_tools = self._init_prompt_tools(tools)
        prompt_messages_tools.extend(self._init_prompt_mcp_tools(mcp_tools))
        self._prompt_messages_tools = prompt_messages_tools
        
        # 打印转换后的工具信息，用于调试
        # print("=== prompt_messages_tools ===")
        # for tool in prompt_messages_tools:
        #     print(f"工具名称: {tool.name}")
        #     print(f"工具描述: {tool.description}")
        #     print(f"工具参数: {tool.parameters}")
        #     print("---")
        # print(f"总共 {len(prompt_messages_tools)} 个工具")
        # print("=============================")

        while run_agent_state and iteration_step <= max_iteration_steps:
            # continue to run until there is not any tool call
            run_agent_state = False
            # 记录当前轮次开始的时间戳，用于计算轮次执行耗时
            round_started_at = time.perf_counter()
            # 打印下第几轮
            print(f"第 {iteration_step} 轮")
            round_log = self.create_log_message(
                label=f"ROUND {iteration_step}",
                data={},
                metadata={
                    LogMetadata.STARTED_AT: round_started_at,
                },
                status=ToolInvokeMessage.LogMessage.LogStatus.START,
            )
            # 将轮次开始的日志消息返回给调用者，用于实时显示执行进度
            yield round_log
            if iteration_step == max_iteration_steps:
                # the last iteration, remove all tools
                self._prompt_messages_tools = []

            message_file_ids: list[str] = []

            # recalc llm max tokens
            prompt_messages = self._organize_prompt_messages(
                agent_scratchpad, self.query, max_iteration_steps
            )
            if model.entity and model.completion_params:
                self.recalc_llm_max_tokens(
                    model.entity, prompt_messages, model.completion_params
                )
            # invoke model
            from datetime import datetime
            print(f"=== 开始调用LLM === {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
            
            # 打印prompt_messages，类似curl格式
            # print("=== LLM API 请求体（类似curl格式）===")
            # api_request = {
            #     "model": model.model,
            #     "messages": [
            #         {
            #             "role": msg.role.value if hasattr(msg, 'role') else (
            #                 "system" if isinstance(msg, SystemPromptMessage) else
            #                 "user" if isinstance(msg, UserPromptMessage) else
            #                 "assistant"
            #             ),
            #             "content": msg.content
            #         }
            #         for msg in prompt_messages
            #     ],
            #     "stream": True,
            #     "stop": stop,
            #     **(model.completion_params if model.completion_params else {})
            # }
            
            # print("curl -X POST 'https://api.openai.com/v1/chat/completions' \\")
            # print("  -H 'Content-Type: application/json' \\")
            # print("  -H 'Authorization: Bearer YOUR_API_KEY' \\")
            # print(f"  -d '{json.dumps(api_request, ensure_ascii=False, indent=2)}'")
            # print("========================================")
            
            chunks = self.session.model.llm.invoke(
                model_config=LLMModelConfig(**model.model_dump(mode="json")),
                prompt_messages=prompt_messages,
                stream=True,
                stop=stop,
            )
            print(f"=== LLM调用完成，开始处理流式输出 === {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

            usage_dict = {}
            # 先将chunks转换为列表以便多次遍历
            chunks_list = list(chunks)
            # print(f"=== 原始chunks数量: {len(chunks_list)} === {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
            # for i, chunk in enumerate(chunks_list[:3]):  # 只打印前3个chunk
            #     print(f"原始chunk {i}: {chunk}")
            
            react_chunks = CotAgentOutputParser.handle_react_stream_output(
                iter(chunks_list), usage_dict  # 重新创建迭代器
            )
            # 也将react_chunks转换为列表
            react_chunks_list = list(react_chunks)
            print(f"=== 解析后react_chunks数量: {len(react_chunks_list)} === {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
            
            scratchpad = AgentScratchpadUnit(
                agent_response="",
                thought="",
                action_str="",
                observation="",
                action=None,
            )

            model_started_at = time.perf_counter()
            model_log = self.create_log_message(
                label=f"{model.model} Thought",
                data={},
                metadata={
                    LogMetadata.STARTED_AT: model_started_at,
                    LogMetadata.PROVIDER: model.provider,
                },
                parent=round_log,
                status=ToolInvokeMessage.LogMessage.LogStatus.START,
            )
            yield model_log

            # 遍历从LLM输出解析出来的ReAct块（思考或动作）
            print(f"=== 开始处理 {len(react_chunks_list)} 个react块 === {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
            
            # 新增变量来收集原始的react_chunks_list所有内容
            original_agent_response = ""
            
            for i, chunk in enumerate(react_chunks_list):
                # print(f"\n--- 处理第 {i+1} 个块 ---")
                
                # 收集原始内容到新变量
                if isinstance(chunk, AgentScratchpadUnit.Action):
                    original_agent_response += json.dumps(chunk.model_dump())
                else:
                    original_agent_response += str(chunk)
                
                # 如果当前块是一个动作(Action)
                if isinstance(chunk, AgentScratchpadUnit.Action):
                    action = chunk
                    print(f"✓ 检测到动作: {action.action_name}")
                    print(f"✓ 动作参数: {action.action_input}")
                    # 检测到动作，将动作信息添加到agent_response中
                    assert scratchpad.agent_response is not None
                    action_json = json.dumps(chunk.model_dump())
                    scratchpad.agent_response += action_json
                    print(f"✓ 添加动作JSON: {action_json}")

                    # 将动作转换为JSON字符串保存，用于后续工具调用
                    scratchpad.action_str = action_json
                    scratchpad.action = action
                    print(f"✓ 动作已保存到scratchpad")
                else:
                    # 如果不是动作，说明是思考过程的文本
                    # print(f"✓ 检测到思考内容: {repr(chunk)}")
                    
                    scratchpad.agent_response = scratchpad.agent_response or ""
                    scratchpad.thought = scratchpad.thought or ""
                    
                    # 将思考内容累加到agent_response和thought中
                    scratchpad.agent_response += str(chunk)
                    scratchpad.thought += str(chunk)
                    
                # print(f"累积思考长度: {len(scratchpad.thought)} 字符")
            
            # 确保thought字段有内容，如果为空就设置默认值
            print(f"\n=== 处理完所有块后 === {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
            print(f"最终思考内容: {repr(scratchpad.thought)}")
            print(f"思考内容长度: {len(scratchpad.thought)}")

            print(f"=== 原始agent_response: {original_agent_response} === {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
            
            scratchpad.thought = (
                scratchpad.thought.strip()
                if scratchpad.thought.strip()  # 确保strip后不为空
                else "I am thinking about how to help you"
            )
            print(f"=== 处理后思考内容: {repr(scratchpad.thought)} === {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
            print(f"=== 是否有动作: {'是' if scratchpad.action else '否'} ===")
            
            # 检查前一轮是否调用了工具
            has_tool_call = False
            tool_name = "无"
            if len(agent_scratchpad) > 0:
                prev_scratchpad = agent_scratchpad[-1]
                has_tool_call = bool(prev_scratchpad.observation and prev_scratchpad.observation.strip())
                tool_name = prev_scratchpad.action.action_name if prev_scratchpad.action else "无"
            
            print(f"=== 前一轮是否调用了工具: {'是' if has_tool_call else '否'} ===")
            print(f"=== 前一轮调用的工具: {tool_name} ===")

            # 1.设置异常最大次数=min(max_iteration_steps,3)
            max_retry_steps = min(max_iteration_steps, 3)
            # 2.如果(是否有动作为否,并且iteration_step<异常最大次数,并且前一轮未调用工具),那么(提问不变并且iteration_step+1),进行新的一轮大模型调用
            if not scratchpad.action and iteration_step < max_retry_steps and not has_tool_call:
                print(f"=== 检测到无动作，当前轮次 {iteration_step} < 最大重试次数 {max_retry_steps}，将进行重试 ===")
                
                # 记录模型调用的日志
                yield self.finish_log_message(
                    log=model_log,
                    data={"thought": scratchpad.thought, "action": {"action": scratchpad.agent_response}},
                    metadata={
                        LogMetadata.STARTED_AT: model_started_at,
                        LogMetadata.FINISHED_AT: time.perf_counter(),
                        LogMetadata.ELAPSED_TIME: time.perf_counter() - model_started_at,
                        LogMetadata.PROVIDER: model.provider,
                        LogMetadata.TOTAL_PRICE: usage_dict["usage"].total_price
                        if usage_dict["usage"]
                        else 0,
                        LogMetadata.CURRENCY: usage_dict["usage"].currency
                        if usage_dict["usage"]
                        else "",
                        LogMetadata.TOTAL_TOKENS: usage_dict["usage"].total_tokens
                        if usage_dict["usage"]
                        else 0,
                    },
                )
                
                # 记录当前轮次的日志
                yield self.finish_log_message(
                    log=round_log,
                    data={
                        "action_name": "",
                        "action_input": "",
                        "thought": scratchpad.thought,
                        "observation": f"无动作，准备重试 (前一轮{'调用了工具' if has_tool_call else '未调用工具'}, 工具: {tool_name})",
                    },
                    metadata={
                        LogMetadata.STARTED_AT: round_started_at,
                        LogMetadata.FINISHED_AT: time.perf_counter(),
                        LogMetadata.ELAPSED_TIME: time.perf_counter() - round_started_at,
                        LogMetadata.TOTAL_PRICE: usage_dict["usage"].total_price
                        if usage_dict["usage"]
                        else 0,
                        LogMetadata.CURRENCY: usage_dict["usage"].currency
                        if usage_dict["usage"]
                        else "",
                        LogMetadata.TOTAL_TOKENS: usage_dict["usage"].total_tokens
                        if usage_dict["usage"]
                        else 0,
                    },
                )
                
                # 重试时修改用户提问，添加指定格式的提示
                self.query = f"Very important! Reminder to ALWAYS respond with a valid json blob of a single action: {self.query}"
                print(f"=== 重试提问: {self.query} ===")
                
                run_agent_state = True
                iteration_step += 1
                continue

            # 将完整的scratchpad添加到历史记录中
            agent_scratchpad.append(scratchpad)
            print(f"=== scratchpad已添加到历史记录，当前历史记录数量: {len(agent_scratchpad)} === {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

            # get llm usage
            if "usage" in usage_dict:
                if usage_dict["usage"] is not None:
                    self.increase_usage(llm_usage, usage_dict["usage"])
            else:
                usage_dict["usage"] = LLMUsage.empty_usage()

            action = (
                scratchpad.action.to_dict()
                if scratchpad.action
                else {"action": scratchpad.agent_response}
            )

            yield self.finish_log_message(
                log=model_log,
                data={"thought": scratchpad.thought, **action},
                metadata={
                    LogMetadata.STARTED_AT: model_started_at,
                    LogMetadata.FINISHED_AT: time.perf_counter(),
                    LogMetadata.ELAPSED_TIME: time.perf_counter() - model_started_at,
                    LogMetadata.PROVIDER: model.provider,
                    LogMetadata.TOTAL_PRICE: usage_dict["usage"].total_price
                    if usage_dict["usage"]
                    else 0,
                    LogMetadata.CURRENCY: usage_dict["usage"].currency
                    if usage_dict["usage"]
                    else "",
                    LogMetadata.TOTAL_TOKENS: usage_dict["usage"].total_tokens
                    if usage_dict["usage"]
                    else 0,
                },
            )
            if not scratchpad.action:
                final_answer = scratchpad.thought
            else:
                if scratchpad.action.action_name.lower() == "final answer":
                    # action is final answer, return final answer directly
                    try:
                        if isinstance(scratchpad.action.action_input, dict):
                            final_answer = json.dumps(scratchpad.action.action_input)
                        elif isinstance(scratchpad.action.action_input, str):
                            final_answer = scratchpad.action.action_input
                        else:
                            final_answer = f"{scratchpad.action.action_input}"
                    except json.JSONDecodeError:
                        final_answer = f"{scratchpad.action.action_input}"
                else:
                    run_agent_state = True
                    # action is tool call, invoke tool
                    tool_call_started_at = time.perf_counter()
                    tool_name = scratchpad.action.action_name
                    tool_call_log = self.create_log_message(
                        label=f"CALL {tool_name}",
                        data={},
                        metadata={
                            LogMetadata.STARTED_AT: time.perf_counter(),
                            LogMetadata.PROVIDER: tool_instances[
                                tool_name
                            ].identity.provider
                            if tool_instances.get(tool_name)
                            else "",
                        },
                        parent=round_log,
                        status=ToolInvokeMessage.LogMessage.LogStatus.START,
                    )
                    yield tool_call_log
                    tool_invoke_response, tool_invoke_parameters = (
                        self._handle_invoke_action(
                            action=scratchpad.action,
                            tool_instances=tool_instances,
                            mcp_clients=mcp_clients,
                            mcp_tool_instances=mcp_tool_instances,
                            message_file_ids=message_file_ids,
                        )
                    )
                    scratchpad.observation = tool_invoke_response
                    scratchpad.agent_response = tool_invoke_response
                    yield self.finish_log_message(
                        log=tool_call_log,
                        data={
                            "tool_name": tool_name,
                            "tool_call_args": tool_invoke_parameters,
                            "output": tool_invoke_response,
                        },
                        metadata={
                            LogMetadata.STARTED_AT: tool_call_started_at,
                            LogMetadata.PROVIDER: tool_instances[
                                tool_name
                            ].identity.provider
                            if tool_instances.get(tool_name)
                            else "",
                            LogMetadata.FINISHED_AT: time.perf_counter(),
                            LogMetadata.ELAPSED_TIME: time.perf_counter()
                                                      - tool_call_started_at,
                        },
                    )

                # update prompt tool message
                for prompt_tool in self._prompt_messages_tools:
                    if prompt_tool.name in tool_instances:
                        self.update_prompt_message_tool(
                            tool_instances[prompt_tool.name], prompt_tool
                        )
            yield self.finish_log_message(
                log=round_log,
                data={
                    "action_name": scratchpad.action.action_name
                    if scratchpad.action
                    else "",
                    "action_input": scratchpad.action.action_input
                    if scratchpad.action
                    else "",
                    "thought": scratchpad.thought,
                    "observation": scratchpad.observation,
                },
                metadata={
                    LogMetadata.STARTED_AT: round_started_at,
                    LogMetadata.FINISHED_AT: time.perf_counter(),
                    LogMetadata.ELAPSED_TIME: time.perf_counter() - round_started_at,
                    LogMetadata.TOTAL_PRICE: usage_dict["usage"].total_price
                    if usage_dict["usage"]
                    else 0,
                    LogMetadata.CURRENCY: usage_dict["usage"].currency
                    if usage_dict["usage"]
                    else "",
                    LogMetadata.TOTAL_TOKENS: usage_dict["usage"].total_tokens
                    if usage_dict["usage"]
                    else 0,
                },
            )
            iteration_step += 1

        # All MCP Client close
        if mcp_clients:
            mcp_clients.close()

        yield self.create_text_message(final_answer)
        yield self.create_json_message(
            {
                "execution_metadata": {
                    LogMetadata.TOTAL_PRICE: llm_usage["usage"].total_price
                    if llm_usage["usage"] is not None
                    else 0,
                    LogMetadata.CURRENCY: llm_usage["usage"].currency
                    if llm_usage["usage"] is not None
                    else "",
                    LogMetadata.TOTAL_TOKENS: llm_usage["usage"].total_tokens
                    if llm_usage["usage"] is not None
                    else 0,
                }
            }
        )

    def _organize_user_query(
            self, query, prompt_messages: list[PromptMessage]
    ) -> list[PromptMessage]:
        """
        组织用户查询消息
        Args:
            query: 用户查询
            prompt_messages: 已有的提示消息列表
        Returns:
            list[PromptMessage]: 更新后的提示消息列表
        """
        prompt_messages.append(UserPromptMessage(content=query))

        return prompt_messages

    def _organize_prompt_messages(
            self, agent_scratchpad: list, query: str, maximum_iterations: int = 5
    ) -> list[PromptMessage]:
        """
        组织完整的提示消息列表
        包含:
        1. 系统提示消息
        2. 历史消息
        3. 用户查询
        4. 助手回复
        
        Args:
            agent_scratchpad: Agent 草稿单元列表
            query: 用户查询
            maximum_iterations: 最大迭代次数，用于动态调整历史记录保留数量
        Returns:
            list[PromptMessage]: 完整的提示消息列表
        """
        # organize system prompt
        system_message = self._system_prompt_message
        
        # organize current assistant messages
        if not agent_scratchpad:
            assistant_messages = []
        else:
            assistant_message = AssistantPromptMessage(content="")
            for unit in agent_scratchpad:
                if unit.is_final():
                    assert isinstance(assistant_message.content, str)
                    assistant_message.content += f"Final Answer: {unit.agent_response}"
                else:
                    assert isinstance(assistant_message.content, str)
                    assistant_message.content += f"Thought: {unit.thought}\n\n"
                    if unit.action_str:
                        # 清理action_input中的具体值以减少噪声, 暂时不清理
                        cleaned_action_str = unit.action_str # self._clean_action_input(unit.action_str)
                        assistant_message.content += f"Action: {cleaned_action_str}\n\n"
                    if unit.observation:
                        assistant_message.content += (
                            f"Observation: {unit.observation}\n\n"
                        )

            assistant_messages = [assistant_message]

        # query messages
        query_messages = self._organize_user_query(query, [])

        if assistant_messages:
            # organize historic prompt messages
            historic_messages = self.history_prompt_messages
            
            messages = [
                system_message,
                *historic_messages,
                *query_messages,
                *assistant_messages,
                UserPromptMessage(content="continue"),
            ]
        else:
            # organize historic prompt messages
            historic_messages = self.history_prompt_messages
            messages = [system_message, *historic_messages, *query_messages]

        # 基于 maximum_iterations 控制最终消息总数
        max_total_messages = min(maximum_iterations, 10)  # 迭代次数，最多10条
        if len(messages) > max_total_messages:
            # 保留system消息（第一条）和最近的消息
            messages = [messages[0]] + messages[-(max_total_messages-1):]
            print(f"=== 控制最终消息数：从原来的更多条减少到 {len(messages)} 条 ===")
        
        print(f"总共 {len(messages)} 条消息")
        print("======================")
        return messages

    def _clean_action_input(self, action_str: str) -> str:
        """
        清理action_str中的action_input值，保留结构但清空具体参数以减少噪声
        Args:
            action_str: 原始的action字符串（JSON格式）
        Returns:
            str: 清理后的action字符串
        """
        try:
            action_data = json.loads(action_str)
            if isinstance(action_data, dict) and "action_input" in action_data:
                # 保留action_input key但清空value
                action_data["action_input"] = {}
                cleaned_str = json.dumps(action_data, ensure_ascii=False)
                print(f"=== 清理action_input: {action_str} -> {cleaned_str} ===")
                return cleaned_str
        except (json.JSONDecodeError, Exception) as e:
            print(f"=== 清理action_input失败: {e}，保持原值 ===")
            
        return action_str

    def _handle_invoke_action(
            self,
            action: AgentScratchpadUnit.Action,
            mcp_clients: McpClients | None,
            tool_instances: Mapping[str, ToolEntity],
            mcp_tool_instances: Mapping[str, dict],
            message_file_ids: list[str],
    ) -> tuple[str, dict[str, Any] | str]:
        """
        处理工具调用动作
        Args:
            action: 要执行的动作
            mcp_clients: MCP 客户端实例
            tool_instances: 工具实例映射
            mcp_tool_instances: MCP 工具实例映射
            message_file_ids: 消息文件ID列表
        Returns:
            tuple: (执行结果, 执行参数)
        """
        # action is tool call, invoke tool
        tool_call_name = action.action_name
        tool_call_args = action.action_input
        tool_instance = tool_instances.get(tool_call_name)
        mcp_tool_instance = mcp_tool_instances.get(tool_call_name)

        if not tool_instance and not mcp_tool_instance:
            answer = f"there is not a tool named {tool_call_name}"
            return answer, tool_call_args

        if isinstance(tool_call_args, str):
            try:
                tool_call_args = json.loads(tool_call_args)
            except json.JSONDecodeError as e:
                params = [
                    param.name
                    for param in tool_instance.parameters
                    if param.form == ToolParameter.ToolParameterForm.LLM
                ]
                if len(params) > 1:
                    raise ValueError("tool call args is not a valid json string") from e
                tool_call_args = {params[0]: tool_call_args} if len(params) == 1 else {}

        tool_invoke_parameters = {}
        try:
            if mcp_tool_instance:
                # invoke MCP tool
                tool_invoke_parameters = tool_call_args
                result = mcp_clients.execute_tool(
                    tool_name=tool_call_name,
                    tool_args=tool_invoke_parameters,
                )
            else:
                # invoke tool
                tool_invoke_parameters = {**tool_instance.runtime_parameters, **tool_call_args}
                tool_invoke_responses = self.session.tool.invoke(
                    provider_type=ToolProviderType(tool_instance.provider_type),
                    provider=tool_instance.identity.provider,
                    tool_name=tool_instance.identity.name,
                    parameters=tool_invoke_parameters,
                )
                result = ""
                for response in tool_invoke_responses:
                    if response.type == ToolInvokeMessage.MessageType.TEXT:
                        result += cast(ToolInvokeMessage.TextMessage, response.message).text
                    elif response.type == ToolInvokeMessage.MessageType.LINK:
                        result += (
                                f"result link: {cast(ToolInvokeMessage.TextMessage, response.message).text}."
                                + " please tell user to check it."
                        )
                    elif response.type in {
                        ToolInvokeMessage.MessageType.IMAGE_LINK,
                        ToolInvokeMessage.MessageType.IMAGE,
                    }:
                        result += (
                                "image has been created and sent to user already, "
                                + "you do not need to create it, just tell the user to check it now."
                        )
                    elif response.type == ToolInvokeMessage.MessageType.JSON:
                        text = json.dumps(
                            cast(
                                ToolInvokeMessage.JsonMessage, response.message
                            ).json_object,
                            ensure_ascii=False,
                        )
                        result += f"tool response: {text}."
                    else:
                        result += f"tool response: {response.message!r}."
        except Exception as e:
            result = f"tool invoke error: {e!s}"

        return result, tool_invoke_parameters

    def _convert_dict_to_action(self, action: dict) -> AgentScratchpadUnit.Action:
        """
        将字典转换为动作对象
        Args:
            action: 动作字典
        Returns:
            AgentScratchpadUnit.Action: 动作对象
        """
        return AgentScratchpadUnit.Action(
            action_name=action["action"], action_input=action["action_input"]
        )

    def _format_assistant_message(
            self, agent_scratchpad: list[AgentScratchpadUnit]
    ) -> str:
        """
        格式化助手消息
        Args:
            agent_scratchpad: Agent 草稿单元列表
        Returns:
            str: 格式化后的消息
        """
        message = ""
        for scratchpad in agent_scratchpad:
            if scratchpad.is_final():
                message += f"Final Answer: {scratchpad.agent_response}"
            else:
                message += f"Thought: {scratchpad.thought}\n\n"
                if scratchpad.action_str:
                    message += f"Action: {scratchpad.action_str}\n\n"
                if scratchpad.observation:
                    message += f"Observation: {scratchpad.observation}\n\n"

        return message

    @staticmethod
    def _init_prompt_mcp_tools(mcp_tools: list[dict]) -> list[PromptMessageTool]:
        """
        初始化 MCP 工具的提示消息
        Args:
            mcp_tools: MCP 工具列表
        Returns:
            list[PromptMessageTool]: 提示消息工具列表
        """
        prompt_messages_tools = []

        for tool in mcp_tools:
            prompt_message = PromptMessageTool(
                name=tool.get("name"),
                description=tool.get("description", ""),
                parameters=tool.get("inputSchema"),
            )
            prompt_messages_tools.append(prompt_message)

        return prompt_messages_tools
