"""Tests for DocumentEngine."""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from core.engines.analysis.document_engine import DocumentEngine
from core.config import Config


@pytest.fixture
def mock_text_engine():
    """Mock TextEngine."""
    engine = MagicMock()
    engine.is_initialized = True
    engine.process = AsyncMock(return_value=("AI response", "model", 100))
    return engine


@pytest.fixture
def document_engine(mock_user_data_manager, mock_text_engine):
    """Create a DocumentEngine with mocked dependencies."""
    engine = DocumentEngine(mock_user_data_manager, mock_text_engine)
    engine.max_file_size_mb = 10
    engine.max_text_reply_chars = 4000
    engine.max_ai_context_chars = 8000
    engine.summary_max_tokens = 300
    return engine


@pytest.mark.asyncio
async def test_process_pdf_success(document_engine):
    """Test successful PDF extraction."""
    pdf_bytes = b"fake pdf content"
    context = {'file_extension': '.pdf', 'caption': ''}

    # Mock _extract_pdf to return text
    document_engine._extract_pdf = AsyncMock(return_value="PDF extracted text")

    result, model, tokens = await document_engine.process(pdf_bytes, context)

    assert result == "PDF extracted text"
    assert model == "document_extraction"
    assert tokens > 0


@pytest.mark.asyncio
async def test_process_docx_success(document_engine):
    """Test successful DOCX extraction."""
    docx_bytes = b"fake docx content"
    context = {'file_extension': '.docx', 'caption': ''}

    document_engine._extract_docx = AsyncMock(return_value="DOCX extracted text")

    result, model, tokens = await document_engine.process(docx_bytes, context)

    assert result == "DOCX extracted text"
    assert model == "document_extraction"


@pytest.mark.asyncio
async def test_process_file_too_large(document_engine):
    """Test file size rejection."""
    large_bytes = b"x" * (11 * 1024 * 1024)  # 11 MB
    context = {'file_extension': '.pdf', 'caption': ''}

    result, model, tokens = await document_engine.process(large_bytes, context)

    assert model == "document_error"
    assert "maximum 10 MB" in result


@pytest.mark.asyncio
async def test_process_unsupported_extension(document_engine):
    """Test unsupported file type."""
    context = {'file_extension': '.txt', 'caption': ''}

    result, model, tokens = await document_engine.process(b"data", context)

    assert model == "document_error"
    assert "Unsupported file type" in result


@pytest.mark.asyncio
async def test_process_with_caption_ai(document_engine, mock_text_engine):
    """Test that caption triggers AI processing."""
    doc_text = "This is a long document text."
    context = {'file_extension': '.pdf', 'caption': 'What is the main idea?', 'user_id': 1}

    document_engine._extract_pdf = AsyncMock(return_value=doc_text)

    result, model, tokens = await document_engine.process(b"data", context)

    # Should call text_engine.process
    mock_text_engine.process.assert_called_once()
    # The prompt should include the document and caption
    call_args = mock_text_engine.process.call_args[0][0]
    assert "What is the main idea?" in call_args
    assert doc_text in call_args
    # model may be "document_ai_model" but we mock text_engine to return "model", but document engine prefixes it
    assert model == "document_ai_model"


@pytest.mark.asyncio
async def test_process_long_text_as_file(document_engine):
    """Test that long text is summarised if text_engine available."""
    long_text = "x" * 5000
    context = {'file_extension': '.pdf', 'caption': ''}

    document_engine._extract_pdf = AsyncMock(return_value=long_text)
    document_engine._generate_summary = AsyncMock(return_value="Summary of long text")

    result, model, tokens = await document_engine.process(b"data", context)

    assert model == "document_summary"
    assert result == "Summary of long text"


@pytest.mark.asyncio
async def test_extract_pdf_with_pypdf_not_installed(document_engine):
    """Test PDF extraction when pypdf is not available."""
    with patch("core.engines.analysis.document_engine.PdfReader", None):
        with pytest.raises(RuntimeError, match="pypdf not installed"):
            await document_engine._extract_pdf(b"data")


@pytest.mark.asyncio
async def test_extract_docx_with_docx_not_installed(document_engine):
    """Test DOCX extraction when python-docx is not available."""
    with patch("core.engines.analysis.document_engine.DocxDocument", None):
        with pytest.raises(RuntimeError, match="python-docx not installed"):
            await document_engine._extract_docx(b"data")