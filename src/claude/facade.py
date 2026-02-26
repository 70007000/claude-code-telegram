"""High-level Claude Code integration facade.

Provides simple interface for bot handlers.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import structlog

from ..config.settings import Settings
from .integration import ClaudeProcessManager, ClaudeResponse, StreamUpdate
from .interactive import InteractiveBridge
from .monitor import ToolMonitor
from .sdk_integration import ClaudeSDKManager
from .session import SessionManager

logger = structlog.get_logger()


class ClaudeIntegration:
    """Main integration point for Claude Code."""

    def __init__(
        self,
        config: Settings,
        process_manager: Optional[ClaudeProcessManager] = None,
        sdk_manager: Optional[ClaudeSDKManager] = None,
        session_manager: Optional[SessionManager] = None,
        tool_monitor: Optional[ToolMonitor] = None,
    ):
        """Initialize Claude integration facade."""
        self.config = config

        # Initialize both managers for fallback capability
        self.sdk_manager = (
            sdk_manager or ClaudeSDKManager(config) if config.use_sdk else None
        )
        self.process_manager = process_manager or ClaudeProcessManager(config)

        # Use SDK by default if configured
        if config.use_sdk:
            self.manager = self.sdk_manager
        else:
            self.manager = self.process_manager

        self.session_manager = session_manager
        self.tool_monitor = tool_monitor
        self.bridge = InteractiveBridge()
        self._sdk_failed_count = 0  # Track SDK failures for adaptive fallback

    async def run_command(
        self,
        prompt: str,
        working_directory: Path,
        user_id: int,
        session_id: Optional[str] = None,
        on_stream: Optional[Callable[[StreamUpdate], None]] = None,
        force_new: bool = False,
        chat_id: Optional[int] = None,
        send_func: Optional[Callable] = None,
    ) -> ClaudeResponse:
        """Run Claude Code command with full integration."""
        logger.info(
            "Running Claude command",
            user_id=user_id,
            working_directory=str(working_directory),
            session_id=session_id,
            prompt_length=len(prompt),
            force_new=force_new,
            chat_id=chat_id,
        )

        # If no session_id provided, try to find an existing session for this
        # user+directory+chat combination (auto-resume).
        # Skip auto-resume when force_new is set (e.g. after /new command).
        if not session_id and not force_new:
            existing_session = await self._find_resumable_session(
                user_id, working_directory, chat_id=chat_id
            )
            if existing_session:
                session_id = existing_session.session_id
                logger.info(
                    "Auto-resuming existing session for project",
                    session_id=session_id,
                    project_path=str(working_directory),
                    user_id=user_id,
                    chat_id=chat_id,
                )

        # Get or create session
        session = await self.session_manager.get_or_create_session(
            user_id, working_directory, session_id, chat_id=chat_id
        )

        # Simple pass-through stream handler
        async def stream_handler(update: StreamUpdate):
            if on_stream:
                try:
                    await on_stream(update)
                except Exception as e:
                    logger.warning("Stream callback failed", error=str(e))

        # Execute command
        try:
            # Continue session if we have a real (non-temporary) session ID
            is_new = getattr(session, "is_new_session", False)
            has_real_session = not is_new and not session.session_id.startswith("temp_")
            should_continue = has_real_session

            # For new sessions, don't pass the temporary session_id to Claude Code
            claude_session_id = session.session_id if has_real_session else None

            try:
                response = await self._execute_with_fallback(
                    prompt=prompt,
                    working_directory=working_directory,
                    session_id=claude_session_id,
                    continue_session=should_continue,
                    stream_callback=stream_handler,
                    chat_id=chat_id,
                    send_func=send_func,
                )
            except Exception as resume_error:
                # If resume failed (e.g., session expired on Claude's side),
                # retry as a fresh session.  The SDK sometimes wraps the real
                # error ("No conversation found") into a generic
                # "Command failed with exit code 1" message, so we also
                # treat any exit-code-1 failure during resume as stale.
                #
                # Also catch context overflow ("prompt is too long",
                # "max.*token", "context.*length", etc.) — these mean the
                # accumulated session history exceeded the model's limit.
                error_lower = str(resume_error).lower()
                is_stale_session = (
                    "no conversation found" in error_lower
                    or "exit code 1" in error_lower
                )
                is_context_overflow = (
                    "prompt is too long" in error_lower
                    or "too long" in error_lower
                    or "max_tokens" in error_lower
                    or "maximum context length" in error_lower
                    or "context_length_exceeded" in error_lower
                    or "request too large" in error_lower
                    or "token limit" in error_lower
                    or "content would exceed" in error_lower
                )
                should_retry_fresh = (
                    (should_continue and is_stale_session)
                    or is_context_overflow
                )
                if should_retry_fresh:
                    reason = "context overflow" if is_context_overflow else "stale session"
                    logger.warning(
                        "Session failed, starting fresh session",
                        reason=reason,
                        failed_session_id=claude_session_id,
                        error=str(resume_error),
                    )
                    # Clean up the broken session
                    await self.session_manager.remove_session(session.session_id)

                    # Create a fresh session and retry
                    session = await self.session_manager.get_or_create_session(
                        user_id, working_directory, chat_id=chat_id
                    )
                    response = await self._execute_with_fallback(
                        prompt=prompt,
                        working_directory=working_directory,
                        session_id=None,
                        continue_session=False,
                        stream_callback=stream_handler,
                        chat_id=chat_id,
                        send_func=send_func,
                    )
                else:
                    raise

            # Update session (this may change the session_id for new sessions)
            old_session_id = session.session_id
            await self.session_manager.update_session(session.session_id, response)

            # For new sessions, get the updated session_id from the session manager
            if hasattr(session, "is_new_session") and response.session_id:
                # The session_id has been updated to Claude's session_id
                final_session_id = response.session_id
            else:
                # Use the original session_id for continuing sessions
                final_session_id = old_session_id

            # Ensure response has the correct session_id
            response.session_id = final_session_id

            logger.info(
                "Claude command completed",
                session_id=response.session_id,
                cost=response.cost,
                duration_ms=response.duration_ms,
                num_turns=response.num_turns,
                is_error=response.is_error,
            )

            return response

        except Exception as e:
            logger.error(
                "Claude command failed",
                error=str(e),
                user_id=user_id,
                session_id=session.session_id,
            )
            raise

    async def _execute_with_fallback(
        self,
        prompt: str,
        working_directory: Path,
        session_id: Optional[str] = None,
        continue_session: bool = False,
        stream_callback: Optional[Callable] = None,
        chat_id: Optional[int] = None,
        send_func: Optional[Callable] = None,
    ) -> ClaudeResponse:
        """Execute command with SDK->subprocess fallback on JSON decode errors."""
        # Try SDK first if configured
        if self.config.use_sdk and self.sdk_manager:
            try:
                logger.debug("Attempting Claude SDK execution")
                response = await self.sdk_manager.execute_command(
                    prompt=prompt,
                    working_directory=working_directory,
                    session_id=session_id,
                    continue_session=continue_session,
                    stream_callback=stream_callback,
                    chat_id=chat_id,
                    send_func=send_func,
                    bridge=self.bridge,
                )
                # Reset failure count on success
                self._sdk_failed_count = 0
                return response

            except Exception as e:
                error_str = str(e)
                # Check if this is a JSON decode error that indicates SDK issues
                if (
                    "Failed to decode JSON" in error_str
                    or "JSON decode error" in error_str
                    or "TaskGroup" in error_str
                    or "ExceptionGroup" in error_str
                    or "Unknown message type" in error_str
                ):
                    self._sdk_failed_count += 1
                    logger.warning(
                        "Claude SDK failed with JSON/TaskGroup error, falling back to subprocess",
                        error=error_str,
                        failure_count=self._sdk_failed_count,
                        error_type=type(e).__name__,
                    )

                    # Use subprocess fallback
                    try:
                        logger.info("Executing with subprocess fallback")
                        # Don't pass SDK session_id to subprocess - start fresh
                        # SDK and subprocess have separate session management
                        response = await self.process_manager.execute_command(
                            prompt=prompt,
                            working_directory=working_directory,
                            session_id=None,  # Start new session in subprocess
                            continue_session=False,  # Fresh start
                            stream_callback=stream_callback,
                        )
                        logger.info("Subprocess fallback succeeded")
                        return response

                    except Exception as fallback_error:
                        logger.error(
                            "Both SDK and subprocess failed",
                            sdk_error=error_str,
                            subprocess_error=str(fallback_error),
                        )
                        # Re-raise the original SDK error since it was the primary method
                        raise e
                else:
                    # For non-JSON errors, re-raise immediately
                    logger.error(
                        "Claude SDK failed with non-JSON error", error=error_str
                    )
                    raise
        else:
            # Use subprocess directly if SDK not configured
            logger.debug("Using subprocess execution (SDK disabled)")
            return await self.process_manager.execute_command(
                prompt=prompt,
                working_directory=working_directory,
                session_id=session_id,
                continue_session=continue_session,
                stream_callback=stream_callback,
            )

    async def _find_resumable_session(
        self,
        user_id: int,
        working_directory: Path,
        chat_id: Optional[int] = None,
    ) -> Optional["ClaudeSession"]:
        """Find the most recent resumable session for a user in a directory+chat.

        Sessions are scoped by (user_id, project_path, chat_id) so that
        1-on-1 and group chats get independent Claude sessions.
        """
        from .session import ClaudeSession

        sessions = await self.session_manager._get_user_sessions(user_id)

        matching_sessions = [
            s
            for s in sessions
            if s.project_path == working_directory
            and not s.session_id.startswith("temp_")
            and not s.is_expired(self.config.session_timeout_hours)
            and s.chat_id == chat_id
        ]

        if not matching_sessions:
            return None

        return max(matching_sessions, key=lambda s: s.last_used)

    async def continue_session(
        self,
        user_id: int,
        working_directory: Path,
        prompt: Optional[str] = None,
        on_stream: Optional[Callable[[StreamUpdate], None]] = None,
    ) -> Optional[ClaudeResponse]:
        """Continue the most recent session."""
        logger.info(
            "Continuing session",
            user_id=user_id,
            working_directory=str(working_directory),
            has_prompt=bool(prompt),
        )

        # Get user's sessions
        sessions = await self.session_manager._get_user_sessions(user_id)

        # Find most recent session in this directory (exclude temporary sessions)
        matching_sessions = [
            s
            for s in sessions
            if s.project_path == working_directory
            and not s.session_id.startswith("temp_")
        ]

        if not matching_sessions:
            logger.info("No matching sessions found", user_id=user_id)
            return None

        # Get most recent
        latest_session = max(matching_sessions, key=lambda s: s.last_used)

        # Continue session with default prompt if none provided
        # Claude CLI requires a prompt, so we use a placeholder
        return await self.run_command(
            prompt=prompt or "Please continue where we left off",
            working_directory=working_directory,
            user_id=user_id,
            session_id=latest_session.session_id,
            on_stream=on_stream,
        )

    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session information."""
        return await self.session_manager.get_session_info(session_id)

    async def get_user_sessions(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all sessions for a user."""
        sessions = await self.session_manager._get_user_sessions(user_id)
        return [
            {
                "session_id": s.session_id,
                "project_path": str(s.project_path),
                "created_at": s.created_at.isoformat(),
                "last_used": s.last_used.isoformat(),
                "total_cost": s.total_cost,
                "message_count": s.message_count,
                "tools_used": s.tools_used,
                "expired": s.is_expired(self.config.session_timeout_hours),
            }
            for s in sessions
        ]

    async def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions."""
        return await self.session_manager.cleanup_expired_sessions()

    async def get_tool_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics."""
        return self.tool_monitor.get_tool_stats()

    async def get_user_summary(self, user_id: int) -> Dict[str, Any]:
        """Get comprehensive user summary."""
        session_summary = await self.session_manager.get_user_session_summary(user_id)
        tool_usage = self.tool_monitor.get_user_tool_usage(user_id)

        return {
            "user_id": user_id,
            **session_summary,
            **tool_usage,
        }

    async def shutdown(self) -> None:
        """Shutdown integration and cleanup resources."""
        logger.info("Shutting down Claude integration")

        # Kill any active processes
        await self.manager.kill_all_processes()

        # Clean up expired sessions
        await self.cleanup_expired_sessions()

        logger.info("Claude integration shutdown complete")

    def _get_admin_instructions(self, blocked_tools: List[str]) -> str:
        """Generate admin instructions for enabling blocked tools."""
        instructions = []

        # Check if settings file exists
        settings_file = Path(".env")

        if blocked_tools:
            # Get current allowed tools and create merged list without duplicates
            current_tools = [
                "Read",
                "Write",
                "Edit",
                "Bash",
                "Glob",
                "Grep",
                "LS",
                "Task",
                "MultiEdit",
                "NotebookRead",
                "NotebookEdit",
                "WebFetch",
                "TodoRead",
                "TodoWrite",
                "WebSearch",
            ]
            merged_tools = list(
                dict.fromkeys(current_tools + blocked_tools)
            )  # Remove duplicates while preserving order
            merged_tools_str = ",".join(merged_tools)
            merged_tools_py = ", ".join(f'"{tool}"' for tool in merged_tools)

            instructions.append("**For Administrators:**")
            instructions.append("")

            if settings_file.exists():
                instructions.append(
                    "To enable these tools, add them to your `.env` file:"
                )
                instructions.append("```")
                instructions.append(f'CLAUDE_ALLOWED_TOOLS="{merged_tools_str}"')
                instructions.append("```")
            else:
                instructions.append("To enable these tools:")
                instructions.append("1. Create a `.env` file in your project root")
                instructions.append("2. Add the following line:")
                instructions.append("```")
                instructions.append(f'CLAUDE_ALLOWED_TOOLS="{merged_tools_str}"')
                instructions.append("```")

            instructions.append("")
            instructions.append("Or modify the default in `src/config/settings.py`:")
            instructions.append("```python")
            instructions.append("claude_allowed_tools: Optional[List[str]] = Field(")
            instructions.append(f"    default=[{merged_tools_py}],")
            instructions.append('    description="List of allowed Claude tools",')
            instructions.append(")")
            instructions.append("```")

        return "\n".join(instructions)

    def _create_tool_error_message(
        self,
        blocked_tools: List[str],
        allowed_tools: List[str],
        admin_instructions: str,
    ) -> str:
        """Create a comprehensive error message for tool validation failures."""
        tool_list = ", ".join(f"`{tool}`" for tool in blocked_tools)
        allowed_list = (
            ", ".join(f"`{tool}`" for tool in allowed_tools)
            if allowed_tools
            else "None"
        )

        message = [
            "🚫 **Tool Access Blocked**",
            "",
            f"Claude tried to use tools that are not currently allowed:",
            f"{tool_list}",
            "",
            "**Why this happened:**",
            "• Claude needs these tools to complete your request",
            "• These tools are not in the allowed tools list",
            "• This is a security feature to control what Claude can do",
            "",
            "**What you can do:**",
            "• Contact the administrator to request access to these tools",
            "• Try rephrasing your request to use different approaches",
            "• Use simpler requests that don't require these tools",
            "",
            "**Currently allowed tools:**",
            f"{allowed_list}",
            "",
            admin_instructions,
        ]

        return "\n".join(message)
