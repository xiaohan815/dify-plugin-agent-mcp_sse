"""
ReAct (Reasoning and Acting) 策略实现
这个模块实现了一个基于ReAct范式的智能代理策略，它结合了推理(Reasoning)和行动(Acting)的能力。
代理通过思考-行动-观察的循环来解决问题。
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
    ReAct策略的参数配置类
    定义了运行ReAct代理所需的所有参数
    """
    query: str  # 用户查询
    instruction: str  # 系统指令
    model: AgentModelConfig  # 模型配置
    tools: list[ToolEntity] | None  # 可用工具列表
    mcp_servers_config: str | None  # MCP服务器配置
    maximum_iterations: int = 3  # 最大迭代次数


class AgentPromptEntity(BaseModel):
    """
    代理提示词实体类
    用于管理代理的提示词模板
    """
    first_prompt: str  # 首次提示词
    next_iteration: str  # 后续迭代的提示词


class ReActAgentStrategy(AgentStrategy):
    """
    ReAct代理策略实现类
    实现了基于ReAct范式的代理策略，通过思考-行动-观察的循环来解决问题
    """
    def __init__(self, runtime, session):
        """
        初始化ReAct代理策略
        Args:
            runtime: 运行时环境
            session: 会话对象
        """
        super().__init__(runtime, session)
        self.query = ""  # 用户查询
        self.instruction = ""  # 系统指令
        self.history_prompt_messages = []  # 历史提示消息
        self.prompt_messages_tools = []  # 工具提示消息

    @property
    def _user_prompt_message(self) -> UserPromptMessage:
        """
        生成用户提示消息
        Returns:
            UserPromptMessage: 用户提示消息对象
        """
        return UserPromptMessage(content=self.query)

    @property
    def _system_prompt_message(self) -> SystemPromptMessage:
        """
        生成系统提示消息
        包含工具列表和指令信息
        Returns:
            SystemPromptMessage: 系统提示消息对象
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

        # 替换提示词模板中的变量
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

        return SystemPromptMessage(content=system_prompt)

    def _invoke(self, parameters: dict[str, Any]) -> Generator[AgentInvokeMessage]:
        """
        运行ReAct代理应用
        实现了ReAct的核心逻辑，包括思考-行动-观察的循环
        
        Args:
            parameters: 运行参数
            
        Yields:
            AgentInvokeMessage: 代理执行过程中的消息
        """

        try:
            react_params = ReActParams(**parameters)
        except pydantic.ValidationError as e:
            raise ValueError(f"Invalid parameters: {e!s}") from e

        # Init parameters
        self.query = react_params.query
        self.instruction = react_params.instruction
        agent_scratchpad = []
        iteration_step = 1
        max_iteration_steps = react_params.maximum_iterations
        run_agent_state = True
        llm_usage: dict[str, Optional[LLMUsage]] = {"usage": None}
        final_answer = ""
        prompt_messages = []

        # Init model
        model = react_params.model
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

        while run_agent_state and iteration_step <= max_iteration_steps:
            # continue to run until there is not any tool call
            run_agent_state = False
            round_started_at = time.perf_counter()
            round_log = self.create_log_message(
                label=f"ROUND {iteration_step}",
                data={},
                metadata={
                    LogMetadata.STARTED_AT: round_started_at,
                },
                status=ToolInvokeMessage.LogMessage.LogStatus.START,
            )
            yield round_log
            if iteration_step == max_iteration_steps:
                # the last iteration, remove all tools
                self._prompt_messages_tools = []

            message_file_ids: list[str] = []

            # recalc llm max tokens
            prompt_messages = self._organize_prompt_messages(
                agent_scratchpad, self.query
            )
            if model.entity and model.completion_params:
                self.recalc_llm_max_tokens(
                    model.entity, prompt_messages, model.completion_params
                )
            # invoke model
            chunks = self.session.model.llm.invoke(
                model_config=LLMModelConfig(**model.model_dump(mode="json")),
                prompt_messages=prompt_messages,
                stream=True,
                stop=stop,
            )

            usage_dict = {}
            react_chunks = CotAgentOutputParser.handle_react_stream_output(
                chunks, usage_dict
            )
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

            for chunk in react_chunks:
                if isinstance(chunk, AgentScratchpadUnit.Action):
                    action = chunk
                    # detect action
                    assert scratchpad.agent_response is not None
                    scratchpad.agent_response += json.dumps(chunk.model_dump())

                    scratchpad.action_str = json.dumps(chunk.model_dump())
                    scratchpad.action = action
                else:
                    scratchpad.agent_response = scratchpad.agent_response or ""
                    scratchpad.thought = scratchpad.thought or ""
                    scratchpad.agent_response += chunk
                    scratchpad.thought += chunk
            scratchpad.thought = (
                scratchpad.thought.strip()
                if scratchpad.thought
                else "I am thinking about how to help you"
            )
            agent_scratchpad.append(scratchpad)

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
        组织用户查询相关的提示消息
        Args:
            query: 用户查询
            prompt_messages: 提示消息列表
        Returns:
            list[PromptMessage]: 组织后的提示消息列表
        """
        prompt_messages.append(UserPromptMessage(content=query))

        return prompt_messages

    def _organize_prompt_messages(
            self, agent_scratchpad: list, query: str
    ) -> list[PromptMessage]:
        """
        组织完整的提示消息列表
        包括系统提示、历史消息、当前思考过程和用户查询
        
        Args:
            agent_scratchpad: 代理的思考过程记录
            query: 用户查询
            
        Returns:
            list[PromptMessage]: 组织后的提示消息列表
        """
        prompt_messages = []
        prompt_messages.append(self._system_prompt_message)
        prompt_messages.extend(self.history_prompt_messages)
        prompt_messages.append(self._user_prompt_message)
        return prompt_messages

    def _handle_invoke_action(
            self,
            action: AgentScratchpadUnit.Action,
            mcp_clients: McpClients | None,
            tool_instances: Mapping[str, ToolEntity],
            mcp_tool_instances: Mapping[str, dict],
            message_file_ids: list[str],
    ) -> tuple[str, dict[str, Any] | str]:
        """
        处理代理的动作调用
        根据动作类型选择合适的工具并执行
        
        Args:
            action: 代理的动作
            mcp_clients: MCP客户端实例
            tool_instances: 工具实例映射
            mcp_tool_instances: MCP工具实例映射
            message_file_ids: 消息文件ID列表
            
        Returns:
            tuple[str, dict[str, Any] | str]: 执行结果和元数据
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
        return AgentScratchpadUnit.Action(**action)

    def _format_assistant_message(
            self, agent_scratchpad: list[AgentScratchpadUnit]
    ) -> str:
        """
        格式化助手的回复消息
        将代理的思考过程转换为可读的文本格式
        
        Args:
            agent_scratchpad: 代理的思考过程记录
            
        Returns:
            str: 格式化后的消息文本
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
        初始化MCP工具的提示消息
        Args:
            mcp_tools: MCP工具列表
        Returns:
            list[PromptMessageTool]: 工具提示消息列表
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
