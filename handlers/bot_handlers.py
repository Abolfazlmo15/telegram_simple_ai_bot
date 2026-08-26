# handlers/bot_handlers.py
from .base_handler import BaseHandler
from .commands.command_handlers import CommandHandlers
from .commands.message_handlers import MessageHandlers
from .commands.mode_handlers import ModeHandlers
from .commands.priority_handlers import PriorityHandlers
from .commands.cancel_handler import CancelHandler

class BotHandlers(
    BaseHandler,
    CommandHandlers,
    MessageHandlers,
    ModeHandlers,
    PriorityHandlers,
    CancelHandler
):
    """
    Combined handlers class.
    Inherits from BaseHandler (which sets all shared attributes)
    and all modular handler classes.
    """
    pass