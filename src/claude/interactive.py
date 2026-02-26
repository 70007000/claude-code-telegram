"""Bridges Claude's interactive prompts (plan mode, questions) with Telegram user input."""

import asyncio
from typing import Any, Callable, Awaitable, Optional

import structlog

logger = structlog.get_logger()


class InteractiveBridge:
    """Bridges Claude's permission requests with Telegram user input.

    When Claude asks for plan approval or user questions, this class:
    1. Sends the options to Telegram as numbered choices
    2. Waits for the user's reply via an asyncio Future
    3. Returns the response to Claude's can_use_tool callback
    """

    def __init__(self) -> None:
        self._waiters: dict[int, asyncio.Future[str]] = {}

    async def wait_for_user(
        self,
        chat_id: int,
        prompt_text: str,
        send_func: Callable[[str], Awaitable[Any]],
        timeout: float = 300,
    ) -> str:
        """Send prompt to Telegram and wait for user response.

        Returns the user's text response, or raises TimeoutError.
        """
        # Cancel any existing waiter for this chat
        if chat_id in self._waiters and not self._waiters[chat_id].done():
            self._waiters[chat_id].cancel()

        loop = asyncio.get_event_loop()
        future: asyncio.Future[str] = loop.create_future()
        self._waiters[chat_id] = future

        # Send the prompt to Telegram
        await send_func(prompt_text)

        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Interactive prompt timed out", chat_id=chat_id)
            raise
        finally:
            self._waiters.pop(chat_id, None)

    def submit(self, chat_id: int, response: str) -> bool:
        """Submit a user response to a waiting prompt. Returns True if consumed."""
        future = self._waiters.get(chat_id)
        if future and not future.done():
            future.set_result(response)
            return True
        return False

    def is_waiting(self, chat_id: int) -> bool:
        """Check if there's a pending prompt for this chat."""
        future = self._waiters.get(chat_id)
        return future is not None and not future.done()
