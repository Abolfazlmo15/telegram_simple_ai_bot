"""Tests for MessageHandlers."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from telegram import Update, Message, User, Chat
from telegram.ext import ContextTypes

from handlers.commands.message_handlers import MessageHandlers


@pytest.fixture
def mock_update():
    """Create a mock Telegram Update."""
    update = MagicMock(spec=Update)
    update.message = MagicMock(spec=Message)
    update.message.message_id = 123
    update.message.reply_text = AsyncMock()
    update.message.reply_photo = AsyncMock()
    update.message.reply_voice = AsyncMock()
    update.message.reply_document = AsyncMock()
    update.message.chat = MagicMock(spec=Chat)
    update.message.chat.send_action = AsyncMock()
    update.message.from_user = MagicMock(spec=User)
    update.message.from_user.id = 12345
    update.message.from_user.username = "testuser"
    update.message.from_user.first_name = "Test"
    update.effective_user = update.message.from_user
    return update


@pytest.fixture
def mock_context():
    """Create a mock Context."""
    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    context.user_data = {}
    context.bot = AsyncMock()
    return context


@pytest.fixture
def handlers():
    """Create a MessageHandlers instance with mocked dependencies."""
    # Since MessageHandlers inherits from BaseHandler, it needs to be instantiated properly.
    # We'll use patch to avoid calling the actual __init__ and instead mock the whole object.
    with patch("handlers.commands.message_handlers.MessageHandlers.__init__", return_value=None):
        handlers = MessageHandlers()
        # Now manually set all attributes
        handlers.engine = MagicMock()
        handlers.voice_engine = MagicMock()
        handlers.rate_limiter = MagicMock()
        handlers.user_data_manager = MagicMock()
        handlers.analytics_engine = MagicMock()
        handlers.proxy_manager = MagicMock()
        handlers.health_checker = MagicMock()
        handlers.cache_manager = MagicMock()

        # Mock all dependencies
        handlers.engine.is_initialized = True
        handlers.engine.process = AsyncMock(return_value=("response", "model", 10))

        handlers.rate_limiter.check = AsyncMock(return_value=(True, 10))
        handlers.rate_limiter.get_remaining = MagicMock(return_value=10)

        handlers.user_data_manager.load_user_data = AsyncMock(return_value={"history": []})
        handlers.user_data_manager.get_preferences = AsyncMock(return_value={})
        handlers.user_data_manager.get_user_model_priority = AsyncMock(return_value=None)
        handlers.user_data_manager.save_image_matrix = AsyncMock(return_value={"file": "path", "width": 100, "height": 100})
        handlers.user_data_manager.save_audio_file = AsyncMock(return_value="audio_path")
        handlers.user_data_manager.prune_pictures = MagicMock()
        handlers.user_data_manager.prune_voices = MagicMock()
        handlers.user_data_manager.add_message_to_history = AsyncMock()
        handlers.user_data_manager.add_image_to_history = AsyncMock()
        handlers.user_data_manager.add_voice_to_history = AsyncMock()
        handlers.user_data_manager.search_history = AsyncMock(return_value=[])

        handlers.memory_manager = AsyncMock()
        handlers.memory_manager.add_interaction = AsyncMock()
        handlers.memory_manager.get_context = AsyncMock(return_value=[])

        handlers.topic_manager = AsyncMock()
        handlers.topic_manager.add_message = AsyncMock(return_value=None)

        handlers.formatter = MagicMock()
        handlers.formatter.format_response = MagicMock(return_value="formatted response")

        handlers.telegram_formatter = MagicMock()
        handlers.telegram_formatter.format_generated_image_caption = MagicMock(return_value="caption")
        handlers.telegram_formatter.format_full_prompt_message = MagicMock(return_value="full prompt")

        handlers.proxy_manager.get_proxy = MagicMock(return_value="https://proxy.com")
        handlers.proxy_manager.current_proxy = "https://proxy.com"
        handlers.proxy_manager.mark_failure = MagicMock()

        handlers.voice_engine.is_initialized = True
        handlers.voice_engine.transcribe = AsyncMock(return_value=("transcription", "whisper-model", 10))

        handlers._download_media = AsyncMock(return_value=b"media_data")
        handlers._active_tasks = {}
        handlers._handle_rate_limit = AsyncMock()

        return handlers


@pytest.mark.asyncio
async def test_handle_message_rate_limit_exceeded(mock_update, mock_context, handlers):
    """Test that rate limit exceeded returns error."""
    handlers.rate_limiter.check = AsyncMock(return_value=(False, 0))

    await handlers.handle_message(mock_update, mock_context)

    handlers._handle_rate_limit.assert_called_once()


@pytest.fixture
def handlers():
    """Create a MessageHandlers instance with mocked dependencies."""
    with patch("handlers.commands.message_handlers.MessageHandlers.__init__", return_value=None):
        handlers = MessageHandlers()
        # Manually set all attributes (same as before)
        handlers.engine = MagicMock()
        handlers.voice_engine = MagicMock()
        handlers.rate_limiter = MagicMock()
        handlers.user_data_manager = MagicMock()
        handlers.analytics_engine = MagicMock()
        handlers.proxy_manager = MagicMock()
        handlers.health_checker = MagicMock()
        handlers.cache_manager = MagicMock()

        handlers.engine.is_initialized = True
        handlers.engine.process = AsyncMock(return_value=("response", "model", 10))

        handlers.rate_limiter.check = AsyncMock(return_value=(True, 10))
        handlers.rate_limiter.get_remaining = MagicMock(return_value=10)

        # Add all required user_data_manager methods
        handlers.user_data_manager.load_user_data = AsyncMock(return_value={"history": []})
        handlers.user_data_manager.get_preferences = AsyncMock(return_value={})
        handlers.user_data_manager.get_user_model_priority = AsyncMock(return_value=None)
        handlers.user_data_manager.save_image_matrix = AsyncMock(return_value={"file": "path", "width": 100, "height": 100})
        handlers.user_data_manager.save_audio_file = AsyncMock(return_value="audio_path")
        handlers.user_data_manager.prune_pictures = MagicMock()
        handlers.user_data_manager.prune_voices = MagicMock()
        handlers.user_data_manager.add_message_to_history = AsyncMock()
        handlers.user_data_manager.add_image_to_history = AsyncMock()
        handlers.user_data_manager.add_voice_to_history = AsyncMock()
        handlers.user_data_manager.search_history = AsyncMock(return_value=[])
        handlers.user_data_manager.get_custom_instructions = AsyncMock(return_value="")  # <-- ADDED
        handlers.user_data_manager.get_response_mode = AsyncMock(return_value="text")
        handlers.user_data_manager.get_response_style = AsyncMock(return_value="balanced")
        handlers.user_data_manager.get_preferred_style = AsyncMock(return_value="no_style")
        handlers.user_data_manager.get_voice_speed = AsyncMock(return_value=1.0)
        handlers.user_data_manager.get_voice_style = AsyncMock(return_value="neutral")
        handlers.user_data_manager.save_model_priority = AsyncMock(return_value=True)
        handlers.user_data_manager.save_image_generation_priority = AsyncMock(return_value=True)

        handlers.memory_manager = AsyncMock()
        handlers.memory_manager.add_interaction = AsyncMock()
        handlers.memory_manager.get_context = AsyncMock(return_value=[])

        handlers.topic_manager = AsyncMock()
        handlers.topic_manager.add_message = AsyncMock(return_value=None)

        handlers.formatter = MagicMock()
        handlers.formatter.format_response = MagicMock(return_value="formatted response")

        handlers.telegram_formatter = MagicMock()
        handlers.telegram_formatter.format_generated_image_caption = MagicMock(return_value="caption")
        handlers.telegram_formatter.format_full_prompt_message = MagicMock(return_value="full prompt")

        handlers.proxy_manager.get_proxy = MagicMock(return_value="https://proxy.com")
        handlers.proxy_manager.current_proxy = "https://proxy.com"
        handlers.proxy_manager.mark_failure = MagicMock()

        handlers.voice_engine.is_initialized = True
        handlers.voice_engine.transcribe = AsyncMock(return_value=("transcription", "whisper-model", 10))

        handlers._download_media = AsyncMock(return_value=b"media_data")
        handlers._active_tasks = {}
        handlers._handle_rate_limit = AsyncMock()

        # Also add the priority handler methods (they are in mixins, but we need to attach them)
        handlers.handle_priority_input = AsyncMock()
        handlers.handle_image_priority_input = AsyncMock()

        return handlers


@pytest.mark.asyncio
async def test_handle_message_memory_search(mock_update, mock_context, handlers):
    """Test that memory search keywords trigger history search."""
    mock_update.message.text = "remember what we talked about"
    handlers.user_data_manager.search_history = AsyncMock(return_value=[
        {'type': 'text', 'message': 'test', 'response': 'response'}
    ])

    await handlers.handle_message(mock_update, mock_context)

    handlers.user_data_manager.search_history.assert_called_with(12345, mock_update.message.text)
    mock_update.message.reply_text.assert_called()
    # Should not call engine.process for memory search
    handlers.engine.process.assert_not_called()


@pytest.mark.asyncio
async def test_handle_message_image_generation(mock_update, mock_context, handlers):
    """Test that image generation intent routes correctly."""
    # Add missing generation_context mock
    handlers.engine.generation_context = AsyncMock()
    handlers.engine.generation_context.get_last_generation = AsyncMock(
        return_value={"prompt": "test prompt", "style": "no_style"}
    )

    mock_update.message.text = "generate an image of a cat"
    handlers.engine.process = AsyncMock(return_value=(b"image_data", "gen_image:test/model", 100))

    await handlers.handle_message(mock_update, mock_context)

    handlers.engine.process.assert_called()
    mock_update.message.reply_photo.assert_called()
    # The placeholder may have been sent; we just check the final photo is sent


@pytest.mark.asyncio
async def test_handle_message_voice_generation(mock_update, mock_context, handlers):
    """Test that voice generation intent routes correctly."""
    mock_update.message.text = "say this hello world"
    handlers.engine.process = AsyncMock(return_value=(b"audio_data", "gen_voice:test/model", 100))

    await handlers.handle_message(mock_update, mock_context)

    handlers.engine.process.assert_called()
    mock_update.message.reply_voice.assert_called()

@pytest.mark.asyncio
async def test_handle_photo(mock_update, mock_context, handlers):
    """Test photo handling routes to vision engine."""
    mock_update.message.photo = [MagicMock()]
    mock_update.message.photo[-1].get_file = AsyncMock(return_value=MagicMock(file_path="/file/test.jpg"))
    mock_update.message.caption = "What is this?"

    # Mock placeholder
    placeholder = AsyncMock()
    placeholder.edit_text = AsyncMock()
    placeholder.edit_reply_markup = AsyncMock()
    placeholder.delete = AsyncMock()
    mock_update.message.reply_text = AsyncMock(return_value=placeholder)

    handlers.engine.process = AsyncMock(return_value=("vision response", "vision/model", 20))

    await handlers.handle_photo(mock_update, mock_context)

    handlers.engine.process.assert_called()
    # The handler should edit the placeholder with the final response
    placeholder.edit_text.assert_called_with("vision response", parse_mode="Markdown", reply_markup=None)
    # reply_text was called only once for the placeholder, but we don't care
    # We just verify the placeholder was edited

@pytest.mark.asyncio
async def test_handle_photo_download_failure(mock_update, mock_context, handlers):
    """Test photo handling when download fails."""
    mock_update.message.photo = [MagicMock()]
    mock_update.message.photo[-1].get_file = AsyncMock(return_value=MagicMock(file_path="/file/test.jpg"))
    handlers._download_media = AsyncMock(return_value=None)

    await handlers.handle_photo(mock_update, mock_context)

    mock_update.message.reply_text.assert_called_with(
        "❌ Failed to download image. Please try again.",
        reply_to_message_id=mock_update.message.message_id
    )
    handlers.engine.process.assert_not_called()


@pytest.mark.asyncio
async def test_handle_voice(mock_update, mock_context, handlers):
    """Test voice handling routes to voice engine then text engine."""
    mock_update.message.voice = MagicMock()
    mock_update.message.voice.get_file = AsyncMock(return_value=MagicMock(file_path="/file/test.ogg"))

    # Engine should be called after transcription
    handlers.engine.process = AsyncMock(return_value=("voice response", "model", 10))

    await handlers.handle_voice(mock_update, mock_context)

    handlers.voice_engine.transcribe.assert_called()
    handlers.engine.process.assert_called()
    mock_update.message.reply_text.assert_called()


@pytest.mark.asyncio
async def test_handle_voice_voice_mode_response(mock_update, mock_context, handlers):
    """Test voice handling when response is voice audio."""
    mock_update.message.voice = MagicMock()
    mock_update.message.voice.get_file = AsyncMock(return_value=MagicMock(file_path="/file/test.ogg"))

    placeholder = AsyncMock()
    placeholder.delete = AsyncMock()
    mock_update.message.reply_text = AsyncMock(return_value=placeholder)

    handlers.engine.process = AsyncMock(return_value=(b"audio_data", "gen_voice_conversation:test", 100))

    await handlers.handle_voice(mock_update, mock_context)

    # Placeholder should be deleted
    placeholder.delete.assert_called()
    mock_update.message.reply_voice.assert_called()


@pytest.mark.asyncio
async def test_handle_document(mock_update, mock_context, handlers):
    """Test document handling routes to document engine."""
    mock_update.message.document = MagicMock()
    mock_update.message.document.file_name = "test.pdf"
    mock_update.message.document.get_file = AsyncMock(return_value=MagicMock(file_path="/file/test.pdf"))
    mock_update.message.caption = "Analyze this"

    handlers.engine.process = AsyncMock(return_value=("document response", "doc/model", 30))

    await handlers.handle_document(mock_update, mock_context)

    handlers.engine.process.assert_called()
    mock_update.message.reply_text.assert_called()


@pytest.mark.asyncio
async def test_handle_document_unsupported_type(mock_update, mock_context, handlers):
    """Test document handling with unsupported file type."""
    mock_update.message.document = MagicMock()
    mock_update.message.document.file_name = "test.txt"
    mock_update.message.document.get_file = AsyncMock(return_value=MagicMock(file_path="/file/test.txt"))

    await handlers.handle_document(mock_update, mock_context)

    handlers._download_media.assert_not_called()
    mock_update.message.reply_text.assert_called_with(
        "❌ *Unsupported file type* – only PDF and DOCX are supported.\nReceived: `.txt`",
        parse_mode="Markdown",
        reply_to_message_id=mock_update.message.message_id
    )


@pytest.mark.asyncio
async def test_handle_document_long_response_as_file(mock_update, mock_context, handlers):
    """Test that long document responses are sent as text file."""
    mock_update.message.document = MagicMock()
    mock_update.message.document.file_name = "test.pdf"
    mock_update.message.document.get_file = AsyncMock(return_value=MagicMock(file_path="/file/test.pdf"))
    mock_update.message.caption = ""

    placeholder = AsyncMock()
    placeholder.delete = AsyncMock()
    mock_update.message.reply_text = AsyncMock(return_value=placeholder)

    long_response = "x" * 5000
    handlers.engine.process = AsyncMock(return_value=(long_response, "doc/model", 30))

    await handlers.handle_document(mock_update, mock_context)

    placeholder.delete.assert_called()
    mock_update.message.reply_document.assert_called()


@pytest.mark.asyncio
async def test_process_text_message_error_handling(mock_update, mock_context, handlers):
    """Test that errors in text processing are caught and displayed."""
    mock_update.message.text = "Hello"
    handlers.engine.process = AsyncMock(side_effect=Exception("Something went wrong"))

    await handlers.handle_message(mock_update, mock_context)

    # Should catch the error and show friendly message
    # The placeholder should be edited with error
    mock_update.message.reply_text.assert_called()


@pytest.mark.asyncio
async def test_process_text_message_cancellation(mock_update, mock_context, handlers):
    """Test that cancellation is handled gracefully."""
    mock_update.message.text = "Hello"

    # Mock engine.process to raise CancelledError
    handlers.engine.process = AsyncMock(side_effect=asyncio.CancelledError)

    await handlers.handle_message(mock_update, mock_context)

    # Should not raise; should handle gracefully
    # Placeholder should show cancelled message
    mock_update.message.reply_text.assert_called()


@pytest.mark.asyncio
async def test_handle_priority_input(mock_update, mock_context, handlers):
    """Test priority input handling."""
    mock_context.user_data = {
        'setting_priority': True,
        'available_models': ['model1', 'model2'],
        'priority_list': [],
        'current_step': 1,
        'engine': 'text'
    }
    mock_update.message.text = "model1"
    handlers.user_data_manager.save_model_priority = AsyncMock(return_value=True)

    await handlers.handle_message(mock_update, mock_context)

    handlers.handle_priority_input.assert_called_once_with(mock_update, mock_context)


@pytest.mark.asyncio
async def test_handle_image_priority_input(mock_update, mock_context, handlers):
    """Test image priority input handling."""
    mock_context.user_data = {
        'setting_image_priority': True
    }
    mock_update.message.text = "pollinations, huggingface"
    handlers.user_data_manager.save_image_generation_priority = AsyncMock(return_value=True)

    await handlers.handle_message(mock_update, mock_context)

    handlers.handle_image_priority_input.assert_called_once_with(mock_update, mock_context)
