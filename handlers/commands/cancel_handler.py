import logging
from telegram import Update, CallbackQuery
from telegram.ext import ContextTypes

logger = logging.getLogger(__name__)


class CancelHandler:
    """Handles the cancel button callback."""

    async def cancel_task(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Generic cancel callback. Expected data: cancel_{user_id}_{message_id}"""
        query: CallbackQuery = update.callback_query
        await query.answer()

        data = query.data
        if not data.startswith("cancel_"):
            return

        try:
            _, user_id_str, msg_id_str = data.split("_")
            user_id = int(user_id_str)
            msg_id = int(msg_id_str)
        except ValueError:
            await query.edit_message_text("❌ Invalid cancellation request.")
            return

        task_key = (user_id, msg_id)
        task = self._active_tasks.pop(task_key, None)

        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Task cancellation error: {e}")

            await query.edit_message_text(
                "🛑 *Processing Cancelled*\n\nYour request has been stopped.",
                parse_mode="Markdown"
            )
        else:
            await query.edit_message_text(
                "✅ *Request Already Completed*\n\nNo action needed.",
                parse_mode="Markdown"
            )