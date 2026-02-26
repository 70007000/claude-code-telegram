"""Claude Code Python SDK integration.

Features:
- Native Claude Code SDK integration
- Async streaming support
- Tool execution management
- Session persistence
"""

import asyncio
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

import structlog
from claude_agent_sdk import (
    AssistantMessage,
    CanUseTool,
    ClaudeAgentOptions,
    ClaudeSDKError,
    CLIConnectionError,
    CLIJSONDecodeError,
    CLINotFoundError,
    Message,
    PermissionResultAllow,
    PermissionResultDeny,
    ProcessError,
    ResultMessage,
    TextBlock,
    ToolPermissionContext,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    query,
)

from .interactive import InteractiveBridge

from ..config.settings import Settings
from .exceptions import (
    ClaudeMCPError,
    ClaudeParsingError,
    ClaudeProcessError,
    ClaudeTimeoutError,
)

logger = structlog.get_logger()


def find_claude_cli(claude_cli_path: Optional[str] = None) -> Optional[str]:
    """Find Claude CLI in common locations."""
    import glob
    import shutil

    # First check if a specific path was provided via config or env
    if claude_cli_path:
        if os.path.exists(claude_cli_path) and os.access(claude_cli_path, os.X_OK):
            return claude_cli_path

    # Check CLAUDE_CLI_PATH environment variable
    env_path = os.environ.get("CLAUDE_CLI_PATH")
    if env_path and os.path.exists(env_path) and os.access(env_path, os.X_OK):
        return env_path

    # Check if claude is already in PATH
    claude_path = shutil.which("claude")
    if claude_path:
        return claude_path

    # Check common installation locations
    common_paths = [
        # NVM installations
        os.path.expanduser("~/.nvm/versions/node/*/bin/claude"),
        # Direct npm global install
        os.path.expanduser("~/.npm-global/bin/claude"),
        os.path.expanduser("~/node_modules/.bin/claude"),
        # System locations
        "/usr/local/bin/claude",
        "/usr/bin/claude",
        # Windows locations (for cross-platform support)
        os.path.expanduser("~/AppData/Roaming/npm/claude.cmd"),
    ]

    for pattern in common_paths:
        matches = glob.glob(pattern)
        if matches:
            # Return the first match
            return matches[0]

    return None


def update_path_for_claude(claude_cli_path: Optional[str] = None) -> bool:
    """Update PATH to include Claude CLI if found."""
    claude_path = find_claude_cli(claude_cli_path)

    if claude_path:
        # Add the directory containing claude to PATH
        claude_dir = os.path.dirname(claude_path)
        current_path = os.environ.get("PATH", "")

        if claude_dir not in current_path:
            os.environ["PATH"] = f"{claude_dir}:{current_path}"
            logger.info("Updated PATH for Claude CLI", claude_path=claude_path)

        return True

    return False


@dataclass
class ClaudeResponse:
    """Response from Claude Code SDK."""

    content: str
    session_id: str
    cost: float
    duration_ms: int
    num_turns: int
    is_error: bool = False
    error_type: Optional[str] = None
    tools_used: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class StreamUpdate:
    """Streaming update from Claude SDK."""

    type: str  # 'assistant', 'user', 'system', 'result'
    content: Optional[str] = None
    tool_calls: Optional[List[Dict]] = None
    metadata: Optional[Dict] = None


class ClaudeSDKManager:
    """Manage Claude Code SDK integration."""

    def __init__(self, config: Settings):
        """Initialize SDK manager with configuration."""
        self.config = config
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

        # Try to find and update PATH for Claude CLI
        if not update_path_for_claude(config.claude_cli_path):
            logger.warning(
                "Claude CLI not found in PATH or common locations. "
                "SDK may fail if Claude is not installed or not in PATH."
            )

        # Set up environment for Claude Code SDK if API key is provided
        # If no API key is provided, the SDK will use existing CLI authentication
        if config.anthropic_api_key_str:
            os.environ["ANTHROPIC_API_KEY"] = config.anthropic_api_key_str
            logger.info("Using provided API key for Claude SDK authentication")
        else:
            logger.info("No API key provided, using existing Claude CLI authentication")

    async def execute_command(
        self,
        prompt: str,
        working_directory: Path,
        session_id: Optional[str] = None,
        continue_session: bool = False,
        stream_callback: Optional[Callable[[StreamUpdate], None]] = None,
        chat_id: Optional[int] = None,
        send_func: Optional[Callable] = None,
        bridge: Optional[InteractiveBridge] = None,
    ) -> ClaudeResponse:
        """Execute Claude Code command via SDK."""
        start_time = asyncio.get_event_loop().time()

        logger.info(
            "Starting Claude SDK command",
            working_directory=str(working_directory),
            session_id=session_id,
            continue_session=continue_session,
        )

        try:
            # Build Claude Agent options
            cli_path = find_claude_cli(self.config.claude_cli_path)

            # Always bypass permissions. The can_use_tool callback pipe is
            # fragile and causes "Stream closed" / "Tool permission request
            # failed" errors that kill the entire session. Single-user bot
            # doesn't need per-tool approval anyway.
            options = ClaudeAgentOptions(
                max_turns=self.config.claude_max_turns,
                cwd=str(working_directory),
                allowed_tools=self.config.claude_allowed_tools,
                permission_mode="bypassPermissions",
                cli_path=cli_path,
                max_buffer_size=10 * 1024 * 1024,  # 10MB (default 1MB too small for MCP)
            )

            # Pass MCP server configuration if enabled
            if self.config.enable_mcp and self.config.mcp_config_path:
                options.mcp_servers = self._load_mcp_config(self.config.mcp_config_path)
                logger.info(
                    "MCP servers configured",
                    mcp_config_path=str(self.config.mcp_config_path),
                )

            # Resume previous session if we have a session_id
            if session_id and continue_session:
                options.resume = session_id
                logger.info(
                    "Resuming previous session",
                    session_id=session_id,
                )

            # Collect messages
            messages = []
            cost = 0.0
            tools_used = []

            # can_use_tool requires AsyncIterable prompt (streaming mode).
            # Wrap the string prompt as a one-shot async generator.
            if options.can_use_tool:
                actual_prompt = self._wrap_prompt_as_stream(prompt, session_id)
            else:
                actual_prompt = prompt

            # Execute with streaming and timeout
            await asyncio.wait_for(
                self._execute_query_with_streaming(
                    actual_prompt, options, messages, stream_callback
                ),
                timeout=self.config.claude_timeout_seconds,
            )

            # Extract cost, tools, and session_id from result message
            cost = 0.0
            tools_used = []
            claude_session_id = None
            for message in messages:
                if isinstance(message, ResultMessage):
                    cost = getattr(message, "total_cost_usd", 0.0) or 0.0
                    claude_session_id = getattr(message, "session_id", None)
                    tools_used = self._extract_tools_from_messages(messages)
                    break

            # Calculate duration
            duration_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

            # Use Claude's session_id if available, otherwise fall back
            final_session_id = claude_session_id or session_id or str(uuid.uuid4())

            if claude_session_id and claude_session_id != session_id:
                logger.info(
                    "Got session ID from Claude",
                    claude_session_id=claude_session_id,
                    previous_session_id=session_id,
                )

            # Update session
            self._update_session(final_session_id, messages)

            return ClaudeResponse(
                content=self._extract_content_from_messages(messages),
                session_id=final_session_id,
                cost=cost,
                duration_ms=duration_ms,
                num_turns=len(
                    [
                        m
                        for m in messages
                        if isinstance(m, (UserMessage, AssistantMessage))
                    ]
                ),
                tools_used=tools_used,
            )

        except asyncio.TimeoutError:
            logger.error(
                "Claude SDK command timed out",
                timeout_seconds=self.config.claude_timeout_seconds,
            )
            raise ClaudeTimeoutError(
                f"Claude SDK timed out after {self.config.claude_timeout_seconds}s"
            )

        except CLINotFoundError as e:
            logger.error("Claude CLI not found", error=str(e))
            error_msg = (
                "Claude Code not found. Please ensure Claude is installed:\n"
                "  npm install -g @anthropic-ai/claude-code\n\n"
                "If already installed, try one of these:\n"
                "  1. Add Claude to your PATH\n"
                "  2. Create a symlink: ln -s $(which claude) /usr/local/bin/claude\n"
                "  3. Set CLAUDE_CLI_PATH environment variable"
            )
            raise ClaudeProcessError(error_msg)

        except ProcessError as e:
            error_str = str(e)
            logger.error(
                "Claude process failed",
                error=error_str,
                exit_code=getattr(e, "exit_code", None),
            )
            # Check if the process error is MCP-related
            if "mcp" in error_str.lower():
                raise ClaudeMCPError(f"MCP server error: {error_str}")
            raise ClaudeProcessError(f"Claude process error: {error_str}")

        except CLIConnectionError as e:
            error_str = str(e)
            logger.error("Claude connection error", error=error_str)
            # Check if the connection error is MCP-related
            if "mcp" in error_str.lower() or "server" in error_str.lower():
                raise ClaudeMCPError(f"MCP server connection failed: {error_str}")
            raise ClaudeProcessError(f"Failed to connect to Claude: {error_str}")

        except CLIJSONDecodeError as e:
            logger.error("Claude SDK JSON decode error", error=str(e))
            raise ClaudeParsingError(f"Failed to decode Claude response: {str(e)}")

        except ClaudeSDKError as e:
            logger.error("Claude SDK error", error=str(e))
            raise ClaudeProcessError(f"Claude SDK error: {str(e)}")

        except Exception as e:
            # Handle ExceptionGroup from TaskGroup operations (Python 3.11+)
            if type(e).__name__ == "ExceptionGroup" or hasattr(e, "exceptions"):
                logger.error(
                    "Task group error in Claude SDK",
                    error=str(e),
                    error_type=type(e).__name__,
                    exception_count=len(getattr(e, "exceptions", [])),
                    exceptions=[
                        str(ex) for ex in getattr(e, "exceptions", [])[:3]
                    ],  # Log first 3 exceptions
                )
                # Extract the most relevant exception from the group
                exceptions = getattr(e, "exceptions", [e])
                main_exception = exceptions[0] if exceptions else e
                raise ClaudeProcessError(
                    f"Claude SDK task error: {str(main_exception)}"
                )

            # Check if it's an ExceptionGroup disguised as a regular exception
            elif hasattr(e, "__notes__") and "TaskGroup" in str(e):
                logger.error(
                    "TaskGroup related error in Claude SDK",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise ClaudeProcessError(f"Claude SDK task error: {str(e)}")

            else:
                logger.error(
                    "Unexpected error in Claude SDK",
                    error=str(e),
                    error_type=type(e).__name__,
                )
                raise ClaudeProcessError(f"Unexpected error: {str(e)}")

    async def _execute_query_with_streaming(
        self, prompt, options, messages: List, stream_callback: Optional[Callable]
    ) -> None:
        """Execute query with streaming and collect messages."""
        try:
            async for message in query(prompt=prompt, options=options):
                messages.append(message)

                # Handle streaming callback
                if stream_callback:
                    try:
                        await self._handle_stream_message(message, stream_callback)
                    except Exception as callback_error:
                        logger.warning(
                            "Stream callback failed",
                            error=str(callback_error),
                            error_type=type(callback_error).__name__,
                        )
                        # Continue processing even if callback fails

        except Exception as e:
            # Handle both ExceptionGroups and regular exceptions
            if type(e).__name__ == "ExceptionGroup" or hasattr(e, "exceptions"):
                logger.error(
                    "TaskGroup error in streaming execution",
                    error=str(e),
                    error_type=type(e).__name__,
                )
            else:
                logger.error(
                    "Error in streaming execution",
                    error=str(e),
                    error_type=type(e).__name__,
                )
            # Re-raise to be handled by the outer try-catch
            raise

    async def _handle_stream_message(
        self, message: Message, stream_callback: Callable[[StreamUpdate], None]
    ) -> None:
        """Handle streaming message from claude-agent-sdk."""
        try:
            if isinstance(message, AssistantMessage):
                # Extract content from assistant message
                content = getattr(message, "content", [])
                text_parts = []
                tool_calls = []

                if content and isinstance(content, list):
                    for block in content:
                        if isinstance(block, ToolUseBlock):
                            tool_calls.append(
                                {
                                    "name": getattr(block, "name", "unknown"),
                                    "input": getattr(block, "input", {}),
                                    "id": getattr(block, "id", None),
                                }
                            )
                        elif hasattr(block, "text"):
                            text_parts.append(block.text)

                if text_parts or tool_calls:
                    update = StreamUpdate(
                        type="assistant",
                        content=("\n".join(text_parts) if text_parts else None),
                        tool_calls=tool_calls if tool_calls else None,
                    )
                    await stream_callback(update)
                elif content:
                    # Fallback for non-list content
                    update = StreamUpdate(
                        type="assistant",
                        content=str(content),
                    )
                    await stream_callback(update)

            elif isinstance(message, UserMessage):
                content = getattr(message, "content", "")
                if content:
                    update = StreamUpdate(
                        type="user",
                        content=content,
                    )
                    await stream_callback(update)

        except Exception as e:
            logger.warning("Stream callback failed", error=str(e))

    def _extract_content_from_messages(self, messages: List[Message]) -> str:
        """Extract content from message list."""
        content_parts = []

        for message in messages:
            if isinstance(message, AssistantMessage):
                content = getattr(message, "content", [])
                if content and isinstance(content, list):
                    # Extract text from TextBlock objects
                    for block in content:
                        if hasattr(block, "text"):
                            content_parts.append(block.text)
                elif content:
                    # Fallback for non-list content
                    content_parts.append(str(content))

        return "\n".join(content_parts)

    def _extract_tools_from_messages(
        self, messages: List[Message]
    ) -> List[Dict[str, Any]]:
        """Extract tools used from message list."""
        tools_used = []
        current_time = asyncio.get_event_loop().time()

        for message in messages:
            if isinstance(message, AssistantMessage):
                content = getattr(message, "content", [])
                if content and isinstance(content, list):
                    for block in content:
                        if isinstance(block, ToolUseBlock):
                            tools_used.append(
                                {
                                    "name": getattr(block, "name", "unknown"),
                                    "timestamp": current_time,
                                    "input": getattr(block, "input", {}),
                                }
                            )

        return tools_used

    def _load_mcp_config(self, config_path: Path) -> Dict[str, Any]:
        """Load MCP server configuration from a JSON file.

        The new claude-agent-sdk expects mcp_servers as a dict, not a file path.
        """
        import json

        try:
            with open(config_path) as f:
                config_data = json.load(f)
            return config_data.get("mcpServers", {})
        except (json.JSONDecodeError, OSError) as e:
            logger.error(
                "Failed to load MCP config", path=str(config_path), error=str(e)
            )
            return {}

    @staticmethod
    async def _wrap_prompt_as_stream(prompt: str, session_id: Optional[str] = None):
        """Wrap a string prompt as an AsyncIterable for streaming mode."""
        yield {
            "type": "user",
            "session_id": session_id or "",
            "message": {"role": "user", "content": prompt},
            "parent_tool_use_id": None,
        }

    def _make_permission_callback(
        self,
        bridge: InteractiveBridge,
        chat_id: int,
        send_func: Callable,
    ) -> CanUseTool:
        """Create a can_use_tool callback that routes interactive tools through Telegram."""

        async def _callback(
            tool_name: str,
            tool_input: Dict[str, Any],
            context: ToolPermissionContext,
        ):
            # Interactive tools: present to user via Telegram
            if tool_name == "ExitPlanMode":
                return await self._handle_exit_plan_mode(
                    tool_input, bridge, chat_id, send_func
                )
            if tool_name == "AskUserQuestion":
                return await self._handle_ask_user_question(
                    tool_input, bridge, chat_id, send_func
                )

            # Everything else: auto-approve
            return PermissionResultAllow()

        return _callback

    async def _handle_exit_plan_mode(
        self,
        tool_input: Dict[str, Any],
        bridge: InteractiveBridge,
        chat_id: int,
        send_func: Callable,
    ):
        """Handle ExitPlanMode: present plan approval options in Telegram."""
        prompt_lines = ["Claude has a plan ready. What would you like to do?\n"]
        prompt_lines.append("1. Approve and start implementation")
        prompt_lines.append("2. Reject / revise the plan")
        prompt_lines.append("\nReply with 1, 2, or type feedback:")

        try:
            response = await bridge.wait_for_user(
                chat_id, "\n".join(prompt_lines), send_func
            )
            response = response.strip()
            if response == "1" or response.lower() in ("yes", "approve", "go", "ok", "y"):
                return PermissionResultAllow()
            else:
                return PermissionResultDeny(
                    message=response if response != "2" else "User rejected the plan.",
                )
        except asyncio.TimeoutError:
            return PermissionResultDeny(message="Timed out waiting for plan approval.")

    async def _handle_ask_user_question(
        self,
        tool_input: Dict[str, Any],
        bridge: InteractiveBridge,
        chat_id: int,
        send_func: Callable,
    ):
        """Handle AskUserQuestion: format question with options, wait for answer."""
        questions = tool_input.get("questions", [])
        if not questions:
            return PermissionResultAllow()

        prompt_lines = []
        for q_idx, q in enumerate(questions):
            question_text = q.get("question", "")
            prompt_lines.append(f"**{question_text}**")
            options = q.get("options", [])
            for i, opt in enumerate(options):
                label = opt.get("label", "")
                desc = opt.get("description", "")
                prompt_lines.append(f"  {i + 1}. {label}" + (f" - {desc}" if desc else ""))
            prompt_lines.append("")

        prompt_lines.append("Reply with a number or type your answer:")

        try:
            response = await bridge.wait_for_user(
                chat_id, "\n".join(prompt_lines), send_func
            )

            # Map numbered response back to option label if possible
            response = response.strip()
            if questions and response.isdigit():
                idx = int(response) - 1
                options = questions[0].get("options", [])
                if 0 <= idx < len(options):
                    # Fill in the answer for the question
                    label = options[idx].get("label", response)
                    answers = {questions[0].get("question", ""): label}
                    return PermissionResultAllow(
                        updated_input={**tool_input, "answers": answers}
                    )

            # Free-text answer
            answers = {questions[0].get("question", ""): response}
            return PermissionResultAllow(
                updated_input={**tool_input, "answers": answers}
            )
        except asyncio.TimeoutError:
            return PermissionResultDeny(message="Timed out waiting for answer.")

    def _update_session(self, session_id: str, messages: List[Message]) -> None:
        """Update session data."""
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = {
                "messages": [],
                "created_at": asyncio.get_event_loop().time(),
            }

        session_data = self.active_sessions[session_id]
        session_data["messages"] = messages
        session_data["last_used"] = asyncio.get_event_loop().time()

    async def kill_all_processes(self) -> None:
        """Kill all active processes (no-op for SDK)."""
        logger.info("Clearing active SDK sessions", count=len(self.active_sessions))
        self.active_sessions.clear()

    def get_active_process_count(self) -> int:
        """Get number of active sessions."""
        return len(self.active_sessions)
