"""Message orchestrator — single entry point for all Telegram updates.

Routes messages based on agentic vs classic mode. In agentic mode, provides
a minimal conversational interface (3 commands, no inline keyboards). In
classic mode, delegates to existing full-featured handlers.
"""

import asyncio
import re
import time
from typing import Any, Callable, Dict, List, Optional

import structlog
from telegram import BotCommand, Update
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from ..claude.exceptions import ClaudeToolValidationError
from ..claude.integration import StreamUpdate
from ..config.settings import Settings
from .message_queue import message_queue
from .utils.html_format import escape_html

logger = structlog.get_logger()

# Patterns that look like secrets/credentials in CLI arguments
_SECRET_PATTERNS: List[re.Pattern[str]] = [
    # API keys / tokens (sk-ant-..., sk-..., ghp_..., gho_..., github_pat_..., xoxb-...)
    re.compile(
        r"(sk-ant-api\d*-[A-Za-z0-9_-]{10})[A-Za-z0-9_-]*"
        r"|(sk-[A-Za-z0-9_-]{20})[A-Za-z0-9_-]*"
        r"|(ghp_[A-Za-z0-9]{5})[A-Za-z0-9]*"
        r"|(gho_[A-Za-z0-9]{5})[A-Za-z0-9]*"
        r"|(github_pat_[A-Za-z0-9_]{5})[A-Za-z0-9_]*"
        r"|(xoxb-[A-Za-z0-9]{5})[A-Za-z0-9-]*"
    ),
    # AWS access keys
    re.compile(r"(AKIA[0-9A-Z]{4})[0-9A-Z]{12}"),
    # Generic long hex/base64 tokens after common flags/env patterns
    re.compile(
        r"((?:--token|--secret|--password|--api-key|--apikey|--auth)"
        r"[= ]+)['\"]?[A-Za-z0-9+/_.:-]{8,}['\"]?"
    ),
    # Inline env assignments like KEY=value
    re.compile(
        r"((?:TOKEN|SECRET|PASSWORD|API_KEY|APIKEY|AUTH_TOKEN|PRIVATE_KEY"
        r"|ACCESS_KEY|CLIENT_SECRET|WEBHOOK_SECRET)"
        r"=)['\"]?[^\s'\"]{8,}['\"]?"
    ),
    # Bearer / Basic auth headers
    re.compile(r"(Bearer )[A-Za-z0-9+/_.:-]{8,}" r"|(Basic )[A-Za-z0-9+/=]{8,}"),
    # Connection strings with credentials  user:pass@host
    re.compile(r"://([^:]+:)[^@]{4,}(@)"),
]


def _redact_secrets(text: str) -> str:
    """Replace likely secrets/credentials with redacted placeholders."""
    result = text
    for pattern in _SECRET_PATTERNS:
        result = pattern.sub(
            lambda m: next((g + "***" for g in m.groups() if g is not None), "***"),
            result,
        )
    return result


# Human-readable tool descriptions for progress display
_TOOL_DISPLAY: Dict[str, str] = {
    "Read": "Reading",
    "Write": "Writing",
    "Edit": "Editing",
    "MultiEdit": "Editing",
    "Bash": "Running",
    "Glob": "Searching",
    "Grep": "Searching",
    "LS": "Listing",
    "Task": "Delegating",
    "WebFetch": "Fetching",
    "WebSearch": "Searching web",
    "NotebookRead": "Reading notebook",
    "NotebookEdit": "Editing notebook",
    "TodoRead": "Reading todos",
    "TodoWrite": "Writing todos",
}


def _tool_display(name: str, detail: str = "") -> str:
    """Return a clean progress line for a tool call."""
    verb = _TOOL_DISPLAY.get(name, name)
    if detail:
        return f"{verb} {detail}"
    return verb


class MessageOrchestrator:
    """Routes messages based on mode. Single entry point for all Telegram updates."""

    def __init__(self, settings: Settings, deps: Dict[str, Any]):
        self.settings = settings
        self.deps = deps

    def _inject_deps(self, handler: Callable) -> Callable:  # type: ignore[type-arg]
        """Wrap handler to inject dependencies into context.bot_data."""

        async def wrapped(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            for key, value in self.deps.items():
                context.bot_data[key] = value
            context.bot_data["settings"] = self.settings
            await handler(update, context)

        return wrapped

    def register_handlers(self, app: Application) -> None:
        """Register handlers based on mode."""
        if self.settings.agentic_mode:
            self._register_agentic_handlers(app)
        else:
            self._register_classic_handlers(app)

    def _register_agentic_handlers(self, app: Application) -> None:
        """Register minimal agentic handlers: commands + text/file/photo."""
        # Commands
        for cmd, handler in [
            ("start", self.agentic_start),
            ("new", self.agentic_new),
            ("stop", self.agentic_stop),
            ("cancel", self.agentic_stop),  # /cancel alias for /stop
            ("status", self.agentic_status),
            ("verbose", self.agentic_verbose),
        ]:
            app.add_handler(CommandHandler(cmd, self._inject_deps(handler)))

        # Text messages -> Claude (via queue)
        app.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self._inject_deps(self.agentic_text_entry),
            ),
            group=10,
        )

        # File uploads -> Claude
        app.add_handler(
            MessageHandler(
                filters.Document.ALL, self._inject_deps(self.agentic_document)
            ),
            group=10,
        )

        # Photo uploads -> Claude
        app.add_handler(
            MessageHandler(filters.PHOTO, self._inject_deps(self.agentic_photo)),
            group=10,
        )

        # Only cd: callbacks (for project selection), scoped by pattern
        app.add_handler(
            CallbackQueryHandler(
                self._inject_deps(self._agentic_callback),
                pattern=r"^cd:",
            )
        )

        logger.info("Agentic handlers registered (commands + text/file/photo)")

    def _register_classic_handlers(self, app: Application) -> None:
        """Register full classic handler set (moved from core.py)."""
        from .handlers import callback, command, message

        handlers = [
            ("start", command.start_command),
            ("help", command.help_command),
            ("new", command.new_session),
            ("continue", command.continue_session),
            ("end", command.end_session),
            ("ls", command.list_files),
            ("cd", command.change_directory),
            ("pwd", command.print_working_directory),
            ("projects", command.show_projects),
            ("status", command.session_status),
            ("export", command.export_session),
            ("actions", command.quick_actions),
            ("git", command.git_command),
        ]

        for cmd, handler in handlers:
            app.add_handler(CommandHandler(cmd, self._inject_deps(handler)))

        app.add_handler(
            MessageHandler(
                filters.TEXT & ~filters.COMMAND,
                self._inject_deps(message.handle_text_message),
            ),
            group=10,
        )
        app.add_handler(
            MessageHandler(
                filters.Document.ALL, self._inject_deps(message.handle_document)
            ),
            group=10,
        )
        app.add_handler(
            MessageHandler(filters.PHOTO, self._inject_deps(message.handle_photo)),
            group=10,
        )
        app.add_handler(
            CallbackQueryHandler(self._inject_deps(callback.handle_callback_query))
        )

        logger.info("Classic handlers registered (13 commands + full handler set)")

    async def get_bot_commands(self) -> list:  # type: ignore[type-arg]
        """Return bot commands appropriate for current mode."""
        if self.settings.agentic_mode:
            return [
                BotCommand("start", "Start the bot"),
                BotCommand("new", "Start a fresh session"),
                BotCommand("stop", "Stop current task"),
                BotCommand("cancel", "Cancel current task and clear queue"),
                BotCommand("status", "Show session status"),
                BotCommand("verbose", "Set output verbosity (0/1/2)"),
            ]
        else:
            return [
                BotCommand("start", "Start bot and show help"),
                BotCommand("help", "Show available commands"),
                BotCommand("new", "Clear context and start fresh session"),
                BotCommand("continue", "Explicitly continue last session"),
                BotCommand("end", "End current session and clear context"),
                BotCommand("ls", "List files in current directory"),
                BotCommand("cd", "Change directory (resumes project session)"),
                BotCommand("pwd", "Show current directory"),
                BotCommand("projects", "Show all projects"),
                BotCommand("status", "Show session status"),
                BotCommand("export", "Export current session"),
                BotCommand("actions", "Show quick actions"),
                BotCommand("git", "Git repository commands"),
            ]

    # --- Agentic handlers ---

    async def agentic_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Brief welcome, no buttons."""
        user = update.effective_user
        current_dir = context.chat_data.get(
            "current_directory", self.settings.approved_directory
        )
        dir_display = f"<code>{current_dir}/</code>"

        safe_name = escape_html(user.first_name)
        await update.message.reply_text(
            f"Hi {safe_name}! I'm your AI coding assistant.\n"
            f"Just tell me what you need — I can read, write, and run code.\n\n"
            f"Working in: {dir_display}\n"
            f"Commands: /new (reset) · /stop · /status",
            parse_mode="HTML",
        )

    async def agentic_new(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Reset session, one-line confirmation."""
        context.chat_data["claude_session_id"] = None
        context.chat_data["session_started"] = True
        context.chat_data["force_new_session"] = True

        await update.message.reply_text("Session reset. What's next?")

    async def agentic_stop(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Cancel the running Claude task, clear queue, kill subprocess."""
        import os
        import signal

        chat_id = update.effective_chat.id

        # Clear queued messages
        cleared = message_queue.cancel_all(chat_id)

        running_task: Optional[asyncio.Task] = context.chat_data.get("running_task")
        if not running_task or running_task.done():
            if cleared:
                await update.message.reply_text(f"Cleared {cleared} queued message(s).")
            else:
                await update.message.reply_text("Nothing running.")
            return

        # Kill child claude processes of our bot PID
        bot_pid = os.getpid()
        try:
            import subprocess

            result = subprocess.run(
                ["pgrep", "-P", str(bot_pid)],
                capture_output=True,
                text=True,
            )
            for child_pid in result.stdout.strip().split("\n"):
                if child_pid:
                    try:
                        os.kill(int(child_pid), signal.SIGTERM)
                    except ProcessLookupError:
                        pass
        except Exception:
            pass

        # Cancel the asyncio task
        running_task.cancel()

        suffix = f" Cleared {cleared} queued." if cleared else ""
        await update.message.reply_text(f"Cancelled.{suffix}")

    async def agentic_status(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Compact one-line status, no buttons."""
        current_dir = context.chat_data.get(
            "current_directory", self.settings.approved_directory
        )
        dir_display = str(current_dir)

        session_id = context.chat_data.get("claude_session_id")
        session_status = "active" if session_id else "none"

        # Cost info
        cost_str = ""
        rate_limiter = context.bot_data.get("rate_limiter")
        if rate_limiter:
            try:
                user_status = rate_limiter.get_user_status(update.effective_user.id)
                cost_usage = user_status.get("cost_usage", {})
                current_cost = cost_usage.get("current", 0.0)
                cost_str = f" · Cost: ${current_cost:.2f}"
            except Exception:
                pass

        # Queue info
        chat_id = update.effective_chat.id
        queue_size = message_queue.queue_size(chat_id)
        queue_str = f" · Queue: {queue_size}" if queue_size else ""

        await update.message.reply_text(
            f"Session: {session_status}{cost_str}{queue_str}\n{dir_display}"
        )

    def _get_verbose_level(self, context: ContextTypes.DEFAULT_TYPE) -> int:
        """Return effective verbose level: per-user override or global default."""
        user_override = context.chat_data.get("verbose_level")
        if user_override is not None:
            return int(user_override)
        return self.settings.verbose_level

    async def agentic_verbose(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Set output verbosity: /verbose [0|1|2]."""
        args = update.message.text.split()[1:] if update.message.text else []
        if not args:
            current = self._get_verbose_level(context)
            labels = {0: "quiet", 1: "normal", 2: "detailed"}
            await update.message.reply_text(
                f"Verbosity: <b>{current}</b> ({labels.get(current, '?')})\n\n"
                "Usage: <code>/verbose 0|1|2</code>\n"
                "  0 = quiet (elapsed time only)\n"
                "  1 = normal (tool names)\n"
                "  2 = detailed (tools with inputs)",
                parse_mode="HTML",
            )
            return

        try:
            level = int(args[0])
            if level not in (0, 1, 2):
                raise ValueError
        except ValueError:
            await update.message.reply_text(
                "Please use: /verbose 0, /verbose 1, or /verbose 2"
            )
            return

        context.chat_data["verbose_level"] = level
        labels = {0: "quiet", 1: "normal", 2: "detailed"}
        await update.message.reply_text(
            f"Verbosity set to <b>{level}</b> ({labels[level]})",
            parse_mode="HTML",
        )

    def _format_verbose_progress(
        self,
        tool_log: List[Dict[str, Any]],
        verbose_level: int,
        start_time: float,
    ) -> str:
        """Build clean progress message: elapsed time + recent tool operations."""
        elapsed = time.time() - start_time

        if verbose_level == 0:
            return f"Working ({elapsed:.0f}s)"

        # Filter to tool entries only (no assistant text in progress)
        tool_entries = [e for e in tool_log if e.get("kind") == "tool"]

        if not tool_entries:
            return f"Working ({elapsed:.0f}s)"

        lines: List[str] = [f"Working ({elapsed:.0f}s)\n"]

        # Show last 5 tool operations
        recent = tool_entries[-5:]
        for i, entry in enumerate(recent):
            name = entry.get("name", "?")
            detail = entry.get("detail", "")
            is_last = i == len(recent) - 1

            if verbose_level >= 2 and detail:
                line = _tool_display(name, detail)
            else:
                line = _tool_display(name, detail)

            # Current (last) operation gets "..." suffix
            if is_last:
                line += "..."

            lines.append(line)

        if len(tool_entries) > 5:
            lines.insert(1, f"({len(tool_entries) - 5} earlier)")

        return "\n".join(lines)

    @staticmethod
    def _summarize_tool_input(tool_name: str, tool_input: Dict[str, Any]) -> str:
        """Return a short human-readable summary of tool input."""
        if not tool_input:
            return ""
        if tool_name in ("Read", "Write", "Edit", "MultiEdit"):
            path = tool_input.get("file_path") or tool_input.get("path", "")
            if path:
                return path.rsplit("/", 1)[-1]
        if tool_name in ("Glob", "Grep"):
            pattern = tool_input.get("pattern", "")
            if pattern:
                return f"'{pattern[:50]}'"
        if tool_name == "Bash":
            cmd = tool_input.get("command", "")
            if cmd:
                return _redact_secrets(cmd[:80])[:60]
        if tool_name in ("WebFetch", "WebSearch"):
            return (tool_input.get("url", "") or tool_input.get("query", ""))[:50]
        if tool_name == "Task":
            desc = tool_input.get("description", "")
            if desc:
                return desc[:50]
        # Generic: show first key's value
        for v in tool_input.values():
            if isinstance(v, str) and v:
                return v[:50]
        return ""

    @staticmethod
    def _start_typing_heartbeat(
        chat: Any,
        interval: float = 2.0,
    ) -> "asyncio.Task[None]":
        """Start a background typing indicator task."""

        async def _heartbeat() -> None:
            try:
                while True:
                    await asyncio.sleep(interval)
                    try:
                        await chat.send_action("typing")
                    except Exception:
                        pass
            except asyncio.CancelledError:
                pass

        return asyncio.create_task(_heartbeat())

    def _make_stream_callback(
        self,
        verbose_level: int,
        progress_msg: Any,
        tool_log: List[Dict[str, Any]],
        start_time: float,
    ) -> Optional[Callable[[StreamUpdate], Any]]:
        """Create a stream callback that updates progress with tool operations.

        Returns a callback for all verbose levels (including 0 for elapsed-only
        updates). Typing indicators are handled by a separate heartbeat task.
        """
        last_edit_time = [0.0]  # mutable container for closure
        # Throttle: 4s for verbose modes, 10s for quiet mode
        throttle_interval = 10.0 if verbose_level == 0 else 4.0

        async def _on_stream(update_obj: StreamUpdate) -> None:
            # Capture tool calls (always, even in quiet mode — needed for final count)
            if update_obj.tool_calls:
                for tc in update_obj.tool_calls:
                    name = tc.get("name", "unknown")
                    detail = self._summarize_tool_input(name, tc.get("input", {}))
                    tool_log.append({"kind": "tool", "name": name, "detail": detail})

            # Throttle progress message edits
            now = time.time()
            if (now - last_edit_time[0]) >= throttle_interval:
                last_edit_time[0] = now
                new_text = self._format_verbose_progress(
                    tool_log, verbose_level, start_time
                )
                try:
                    await progress_msg.edit_text(new_text)
                except Exception:
                    pass

        return _on_stream

    async def agentic_text_entry(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Entry point for text messages — routes through per-chat queue."""
        chat_id = update.effective_chat.id

        await message_queue.enqueue_or_run(
            chat_id=chat_id,
            handler=self._agentic_text_impl,
            args=(update, context),
            reply_func=update.message.reply_text,
        )

    async def _agentic_text_impl(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Process a text message with Claude. Called by queue."""
        user_id = update.effective_user.id
        chat_id = update.effective_chat.id
        message_text = update.message.text

        logger.info(
            "Agentic text message",
            user_id=user_id,
            chat_id=chat_id,
            message_length=len(message_text),
        )

        # Rate limit check
        rate_limiter = context.bot_data.get("rate_limiter")
        if rate_limiter:
            allowed, limit_message = await rate_limiter.check_rate_limit(user_id, 0.001)
            if not allowed:
                await update.message.reply_text(f"Rate limited: {limit_message}")
                return

        chat = update.message.chat
        await chat.send_action("typing")

        verbose_level = self._get_verbose_level(context)
        progress_msg = await update.message.reply_text("Working...")

        claude_integration = context.bot_data.get("claude_integration")
        if not claude_integration:
            await progress_msg.edit_text(
                "Claude integration not available. Check configuration."
            )
            return

        current_dir = context.chat_data.get(
            "current_directory", self.settings.approved_directory
        )
        session_id = context.chat_data.get("claude_session_id")

        # Check if /new was used — skip auto-resume for this first message
        force_new = bool(context.chat_data.get("force_new_session"))

        # --- Progress tracking via stream callback ---
        tool_log: List[Dict[str, Any]] = []
        start_time = time.time()
        on_stream = self._make_stream_callback(
            verbose_level, progress_msg, tool_log, start_time
        )

        # Independent typing heartbeat
        heartbeat = self._start_typing_heartbeat(chat)

        # Wrap in a task so /stop can cancel it
        chat_id = update.effective_chat.id

        async def _run_claude():
            return await claude_integration.run_command(
                prompt=message_text,
                working_directory=current_dir,
                user_id=user_id,
                session_id=session_id,
                on_stream=on_stream,
                force_new=force_new,
                chat_id=chat_id,
            )

        task = asyncio.create_task(_run_claude())
        context.chat_data["running_task"] = task

        success = True
        try:
            claude_response = await task

            # Clear force_new flag after successful run
            if force_new:
                context.chat_data["force_new_session"] = False

            context.chat_data["claude_session_id"] = claude_response.session_id

            # Track directory changes
            from .handlers.message import _update_working_directory_from_claude_response

            _update_working_directory_from_claude_response(
                claude_response, context, self.settings, user_id
            )

            # Store interaction
            storage = context.bot_data.get("storage")
            if storage:
                try:
                    await storage.save_claude_interaction(
                        user_id=user_id,
                        session_id=claude_response.session_id,
                        prompt=message_text,
                        response=claude_response,
                        ip_address=None,
                    )
                except Exception as e:
                    logger.warning("Failed to log interaction", error=str(e))

            # Format response
            from .utils.formatting import ResponseFormatter

            formatter = ResponseFormatter(self.settings)
            formatted_messages = formatter.format_claude_response(
                claude_response.content
            )

        except asyncio.CancelledError:
            success = False
            from .utils.formatting import FormattedMessage

            formatted_messages = [FormattedMessage("Cancelled.")]

        except ClaudeToolValidationError as e:
            success = False
            logger.error("Tool validation error", error=str(e), user_id=user_id)
            from .utils.formatting import FormattedMessage

            formatted_messages = [FormattedMessage(str(e), parse_mode="HTML")]

        except Exception as e:
            success = False
            logger.error("Claude integration failed", error=str(e), user_id=user_id)
            from .handlers.message import _format_error_message
            from .utils.formatting import FormattedMessage

            formatted_messages = [
                FormattedMessage(_format_error_message(str(e)), parse_mode="HTML")
            ]
        finally:
            context.chat_data.pop("running_task", None)
            heartbeat.cancel()

        # --- Send response ---
        # If single message that fits, edit the progress message (no bubble jump)
        # If multiple messages needed, delete progress and send new ones
        await self._send_response(update, progress_msg, formatted_messages)

        # Audit log
        audit_logger = context.bot_data.get("audit_logger")
        if audit_logger:
            await audit_logger.log_command(
                user_id=user_id,
                command="text_message",
                args=[message_text[:100]],
                success=success,
            )

    async def _send_response(
        self,
        update: Update,
        progress_msg: Any,
        formatted_messages: List[Any],
    ) -> None:
        """Send formatted response, editing progress msg when possible."""
        if not formatted_messages:
            try:
                await progress_msg.delete()
            except Exception:
                pass
            return

        # Single message that fits -> edit progress message in-place (no bubble jump)
        if len(formatted_messages) == 1 and len(formatted_messages[0].text) <= 4000:
            msg = formatted_messages[0]
            try:
                await progress_msg.edit_text(
                    msg.text,
                    parse_mode=msg.parse_mode,
                )
                return
            except Exception:
                # Fallback: if edit fails (e.g. parse error), delete and send new
                pass

        # Multiple messages or edit failed -> delete progress, send new messages
        try:
            await progress_msg.delete()
        except Exception:
            pass

        for i, message in enumerate(formatted_messages):
            try:
                await update.message.reply_text(
                    message.text,
                    parse_mode=message.parse_mode,
                    reply_markup=None,
                    reply_to_message_id=(update.message.message_id if i == 0 else None),
                )
                if i < len(formatted_messages) - 1:
                    await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning(
                    "Failed to send HTML response, retrying as plain text",
                    error=str(e),
                    message_index=i,
                )
                try:
                    await update.message.reply_text(
                        message.text,
                        reply_markup=None,
                        reply_to_message_id=(
                            update.message.message_id if i == 0 else None
                        ),
                    )
                except Exception:
                    await update.message.reply_text(
                        "Failed to send response. Please try again.",
                        reply_to_message_id=(
                            update.message.message_id if i == 0 else None
                        ),
                    )

    async def agentic_document(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Process file upload -> Claude, minimal chrome."""
        user_id = update.effective_user.id
        document = update.message.document

        logger.info(
            "Agentic document upload",
            user_id=user_id,
            filename=document.file_name,
        )

        # Security validation
        security_validator = context.bot_data.get("security_validator")
        if security_validator:
            valid, error = security_validator.validate_filename(document.file_name)
            if not valid:
                await update.message.reply_text(f"File rejected: {error}")
                return

        # Size check
        max_size = 10 * 1024 * 1024
        if document.file_size > max_size:
            await update.message.reply_text(
                f"File too large ({document.file_size / 1024 / 1024:.1f}MB). Max: 10MB."
            )
            return

        chat = update.message.chat
        await chat.send_action("typing")
        progress_msg = await update.message.reply_text("Working...")

        # Try enhanced file handler, fall back to basic
        features = context.bot_data.get("features")
        file_handler = features.get_file_handler() if features else None
        prompt: Optional[str] = None

        if file_handler:
            try:
                processed_file = await file_handler.handle_document_upload(
                    document,
                    user_id,
                    update.message.caption or "Please review this file:",
                )
                prompt = processed_file.prompt
            except Exception:
                file_handler = None

        if not file_handler:
            file = await document.get_file()
            file_bytes = await file.download_as_bytearray()
            try:
                content = file_bytes.decode("utf-8")
                if len(content) > 50000:
                    content = content[:50000] + "\n... (truncated)"
                caption = update.message.caption or "Please review this file:"
                prompt = (
                    f"{caption}\n\n**File:** `{document.file_name}`\n\n"
                    f"```\n{content}\n```"
                )
            except UnicodeDecodeError:
                await progress_msg.edit_text(
                    "Unsupported file format. Must be text-based (UTF-8)."
                )
                return

        # Process with Claude
        claude_integration = context.bot_data.get("claude_integration")
        if not claude_integration:
            await progress_msg.edit_text(
                "Claude integration not available. Check configuration."
            )
            return

        current_dir = context.chat_data.get(
            "current_directory", self.settings.approved_directory
        )
        session_id = context.chat_data.get("claude_session_id")
        force_new = bool(context.chat_data.get("force_new_session"))

        verbose_level = self._get_verbose_level(context)
        tool_log: List[Dict[str, Any]] = []
        on_stream = self._make_stream_callback(
            verbose_level, progress_msg, tool_log, time.time()
        )

        heartbeat = self._start_typing_heartbeat(chat)
        try:
            chat_id = update.effective_chat.id
            claude_response = await claude_integration.run_command(
                prompt=prompt,
                working_directory=current_dir,
                user_id=user_id,
                session_id=session_id,
                on_stream=on_stream,
                force_new=force_new,
                chat_id=chat_id,
            )

            if force_new:
                context.chat_data["force_new_session"] = False

            context.chat_data["claude_session_id"] = claude_response.session_id

            from .handlers.message import _update_working_directory_from_claude_response

            _update_working_directory_from_claude_response(
                claude_response, context, self.settings, user_id
            )

            from .utils.formatting import ResponseFormatter

            formatter = ResponseFormatter(self.settings)
            formatted_messages = formatter.format_claude_response(
                claude_response.content
            )

            await self._send_response(update, progress_msg, formatted_messages)

        except Exception as e:
            from .handlers.message import _format_error_message

            await progress_msg.edit_text(
                _format_error_message(str(e)), parse_mode="HTML"
            )
            logger.error("Claude file processing failed", error=str(e), user_id=user_id)
        finally:
            heartbeat.cancel()

    async def agentic_photo(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Process photo -> Claude, minimal chrome."""
        user_id = update.effective_user.id

        features = context.bot_data.get("features")
        image_handler = features.get_image_handler() if features else None

        if not image_handler:
            await update.message.reply_text("Photo processing is not available.")
            return

        chat = update.message.chat
        await chat.send_action("typing")
        progress_msg = await update.message.reply_text("Working...")

        try:
            photo = update.message.photo[-1]
            processed_image = await image_handler.process_image(
                photo, update.message.caption
            )

            claude_integration = context.bot_data.get("claude_integration")
            if not claude_integration:
                await progress_msg.edit_text(
                    "Claude integration not available. Check configuration."
                )
                return

            current_dir = context.chat_data.get(
                "current_directory", self.settings.approved_directory
            )
            session_id = context.chat_data.get("claude_session_id")
            force_new = bool(context.chat_data.get("force_new_session"))

            verbose_level = self._get_verbose_level(context)
            tool_log: List[Dict[str, Any]] = []
            on_stream = self._make_stream_callback(
                verbose_level, progress_msg, tool_log, time.time()
            )

            chat_id = update.effective_chat.id
            heartbeat = self._start_typing_heartbeat(chat)
            try:
                claude_response = await claude_integration.run_command(
                    prompt=processed_image.prompt,
                    working_directory=current_dir,
                    user_id=user_id,
                    session_id=session_id,
                    on_stream=on_stream,
                    force_new=force_new,
                    chat_id=chat_id,
                )
            finally:
                heartbeat.cancel()

            if force_new:
                context.chat_data["force_new_session"] = False

            context.chat_data["claude_session_id"] = claude_response.session_id

            from .utils.formatting import ResponseFormatter

            formatter = ResponseFormatter(self.settings)
            formatted_messages = formatter.format_claude_response(
                claude_response.content
            )

            await self._send_response(update, progress_msg, formatted_messages)

        except Exception as e:
            from .handlers.message import _format_error_message

            await progress_msg.edit_text(
                _format_error_message(str(e)), parse_mode="HTML"
            )
            logger.error(
                "Claude photo processing failed", error=str(e), user_id=user_id
            )

    async def _agentic_callback(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle cd: callbacks (pattern-filtered by registration)."""
        query = update.callback_query
        await query.answer()

        data = query.data
        _, param = data.split(":", 1)

        from .handlers.callback import handle_cd_callback

        await handle_cd_callback(query, param, context)
