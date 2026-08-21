"""Base engine that routes to appropriate engine based on input type."""
import logging
from typing import Dict, Tuple, Optional, Any, Union
from PIL import Image
import io

logger = logging.getLogger(__name__)


class BaseEngine:
    def __init__(self, user_data_manager):
        self.user_data_manager = user_data_manager
        self.text_engine = None
        self.vision_engine = None
        self.is_initialized = False
        logger.info("Base Engine (Router) initialized")

    async def initialize(self) -> bool:
        try:
            logger.info("Initializing all engines...")
            from core.engines.text_engine import TextEngine
            from core.engines.vision_engine import VisionEngine

            self.text_engine = TextEngine(self.user_data_manager)
            text_success = await self.text_engine.initialize()
            if not text_success:
                logger.error("Failed to initialize text engine")
                return False

            self.vision_engine = VisionEngine(self.user_data_manager)
            vision_success = await self.vision_engine.initialize()
            if not vision_success:
                logger.warning("Vision engine failed to initialize, continuing without it")

            self.is_initialized = True
            logger.info("✅ All engines initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize engines: {e}", exc_info=True)
            return False

    async def shutdown(self) -> None:
        logger.info("Shutting down all engines...")
        if self.text_engine:
            await self.text_engine.shutdown()
        if self.vision_engine:
            await self.vision_engine.shutdown()
        self.is_initialized = False
        logger.info("All engines shutdown complete")

    async def process(self, input_data: Any, context: Optional[Dict] = None) -> Tuple[str, str, int]:
        if not self.is_initialized:
            raise RuntimeError("Base engine not initialized. Call initialize() first.")
        try:
            if isinstance(input_data, str):
                logger.debug("Routing to TextEngine")
                return await self.text_engine.process(input_data, context)
            elif isinstance(input_data, bytes):
                logger.debug("Routing to VisionEngine (bytes)")
                return await self.vision_engine.process(input_data, context)
            elif isinstance(input_data, Image.Image):
                logger.debug("Routing to VisionEngine (PIL Image)")
                return await self.vision_engine.process(input_data, context)
            else:
                logger.warning(f"Unknown input type {type(input_data)}, treating as text")
                return await self.text_engine.process(str(input_data), context)
        except Exception as e:
            logger.error(f"Error in base engine routing: {e}", exc_info=True)
            raise

    def get_engine_info(self) -> Dict[str, Any]:
        return {
            "base_engine": {
                "initialized": self.is_initialized,
                "type": "Router"
            },
            "text_engine": self.text_engine.get_engine_info() if self.text_engine else None,
            "vision_engine": self.vision_engine.get_engine_info() if self.vision_engine else None
        }