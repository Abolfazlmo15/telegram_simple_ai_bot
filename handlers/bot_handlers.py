# handlers/bot_handlers.py
from .base_handler import BaseHandler
from .command_handlers import CommandHandlers
from .message_handlers import MessageHandlers
from .mode_handlers import ModeHandlers
from .priority_handlers import PriorityHandlers
from .cancel_handler import CancelHandler

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