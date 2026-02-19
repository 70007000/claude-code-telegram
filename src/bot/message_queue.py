"""Per-chat message queue to prevent race conditions.

When a user sends a message while the bot is already processing one,
the new message is queued and processed after the current task finishes.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Dict, Optional

import structlog

logger = structlog.get_logger()


@dataclass
class QueuedMessage:
    """A message waiting to be processed."""

    handler: Callable[..., Coroutine]
    args: tuple
    kwargs: dict
    reply_func: Optional[Callable] = None  # to send "Queued" reply


class ChatMessageQueue:
    """Per-chat FIFO queue ensuring one Claude task at a time per chat."""

    def __init__(self) -> None:
        self._processing: Dict[int, bool] = {}
        self._queues: Dict[int, asyncio.Queue] = {}
        self._current_tasks: Dict[int, asyncio.Task] = {}

    def _get_queue(self, chat_id: int) -> asyncio.Queue:
        if chat_id not in self._queues:
            self._queues[chat_id] = asyncio.Queue()
        return self._queues[chat_id]

    def is_busy(self, chat_id: int) -> bool:
        return self._processing.get(chat_id, False)

    def queue_size(self, chat_id: int) -> int:
        if chat_id not in self._queues:
            return 0
        return self._queues[chat_id].qsize()

    async def enqueue_or_run(
        self,
        chat_id: int,
        handler: Callable[..., Coroutine],
        args: tuple = (),
        kwargs: Optional[dict] = None,
        reply_func: Optional[Callable] = None,
    ) -> None:
        """Run handler immediately if idle, or queue it if busy.

        Args:
            chat_id: Telegram chat ID
            handler: Async function to call
            args: Positional args for handler
            kwargs: Keyword args for handler
            reply_func: Async callable to send "Queued" notification
        """
        kwargs = kwargs or {}

        if self.is_busy(chat_id):
            # Queue the message
            q = self._get_queue(chat_id)
            q.put_nowait(QueuedMessage(handler, args, kwargs, reply_func))

            pos = q.qsize()
            logger.info("Message queued", chat_id=chat_id, position=pos)

            if reply_func:
                try:
                    suffix = f" ({pos} ahead)" if pos > 1 else ""
                    await reply_func(f"Queued{suffix} — I'll get to this next.")
                except Exception:
                    pass
            return

        # Not busy — run immediately and then drain queue
        await self._run_and_drain(chat_id, handler, args, kwargs)

    async def _run_and_drain(
        self,
        chat_id: int,
        handler: Callable[..., Coroutine],
        args: tuple,
        kwargs: dict,
    ) -> None:
        """Run a handler, then drain any queued messages."""
        self._processing[chat_id] = True
        try:
            await handler(*args, **kwargs)
        except Exception as e:
            logger.error("Handler failed", chat_id=chat_id, error=str(e))
        finally:
            self._processing[chat_id] = False

        # Drain queue
        await self._drain_queue(chat_id)

    async def _drain_queue(self, chat_id: int) -> None:
        """Process next queued message if any."""
        q = self._get_queue(chat_id)
        if q.empty():
            return

        msg: QueuedMessage = q.get_nowait()
        logger.info(
            "Processing queued message",
            chat_id=chat_id,
            remaining=q.qsize(),
        )
        await self._run_and_drain(chat_id, msg.handler, msg.args, msg.kwargs)

    def cancel_all(self, chat_id: int) -> int:
        """Clear all queued messages for a chat. Returns count cleared."""
        q = self._get_queue(chat_id)
        count = 0
        while not q.empty():
            try:
                q.get_nowait()
                count += 1
            except asyncio.QueueEmpty:
                break
        return count


# Global singleton
message_queue = ChatMessageQueue()
