"""
Core template manager that handles fetching, caching, and applying templates.
Manages both GitHub-sourced and local fallback templates.
"""
import logging
import json
import asyncio
import random
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path

from prompt_engineering.templates.github_fetcher import GithubFetcher
from prompt_engineering.templates.local_templates import LocalTemplates

logger = logging.getLogger(__name__)


class TemplateManager:
    """
    Manages prompt templates for image generation.

    Features:
    - Fetches templates from GitHub repositories
    - Caches templates locally for fast access
    - Falls back to local templates if GitHub is unavailable
    - Auto-updates every N minutes (configurable)
    - Provides style-specific templates
    - In-memory TTL-based caching to reduce disk reads
    """

    def __init__(self, cache_dir: str = "prompt_cache", update_interval_minutes: int = 10):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.update_interval = timedelta(minutes=update_interval_minutes)
        self._templates_cache: Dict[str, Any] = {}
        self._last_update: Optional[datetime] = None
        self._github_fetcher = GithubFetcher()
        self._local_templates = LocalTemplates()
        self._is_initialized = False
        self._lock = asyncio.Lock()
        # TTL cache for rapid access
        self._memory_ttl = timedelta(minutes=5)
        self._last_memory_update: Optional[datetime] = None
        logger.info(f"📋 TemplateManager initialized (cache: {cache_dir}, interval: {update_interval_minutes}min)")

    async def initialize(self) -> bool:
        """Initialize the template manager by loading templates."""
        async with self._lock:
            if self._is_initialized and self._last_update and (datetime.now() - self._last_update) < self.update_interval:
                return True

            # Try to load from cache file first
            if await self._load_from_cache():
                self._is_initialized = True
                self._last_memory_update = datetime.now()
                logger.info("✅ Templates loaded from cache")
                return True

            # If cache is empty or outdated, fetch from GitHub or use local fallback
            success = await self._fetch_and_cache()
            if success:
                self._is_initialized = True
                self._last_memory_update = datetime.now()
                return True

            # Fallback to local templates
            self._templates_cache = self._local_templates.get_all_templates()
            self._is_initialized = True
            self._last_memory_update = datetime.now()
            logger.info("⚠️ Using local templates (GitHub unavailable)")
            return True

    async def _load_from_cache(self) -> bool:
        """Load templates from local cache file if it exists and is not outdated."""
        cache_file = self.cache_dir / "templates_cache.json"
        if not cache_file.exists():
            return False

        try:
            # Check file modification time
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - mtime > self.update_interval:
                logger.info("Cache file is outdated, will refresh")
                return False

            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            timestamp = datetime.fromisoformat(data.get('timestamp', ''))
            if datetime.now() - timestamp > self.update_interval:
                logger.info("Cache timestamp is outdated, will refresh")
                return False

            self._templates_cache = data.get('templates', {})
            self._last_update = timestamp
            logger.info(f"✅ Loaded {len(self._templates_cache)} templates from cache")
            return True
        except Exception as e:
            logger.error(f"Failed to load cache: {e}")
            return False

    async def _save_to_cache(self) -> bool:
        """Save templates to local cache file."""
        cache_file = self.cache_dir / "templates_cache.json"
        try:
            data = {
                'timestamp': datetime.now().isoformat(),
                'templates': self._templates_cache
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Saved {len(self._templates_cache)} templates to cache")
            return True
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
            return False

    async def _fetch_and_cache(self) -> bool:
        """Fetch templates from GitHub and save to cache."""
        try:
            templates = await self._github_fetcher.fetch_templates()
            if templates:
                self._templates_cache = templates
                self._last_update = datetime.now()
                await self._save_to_cache()
                logger.info(f"✅ Fetched {len(templates)} templates from GitHub")
                return True
        except Exception as e:
            logger.error(f"Failed to fetch from GitHub: {e}")
        return False

    async def refresh(self) -> bool:
        """Manually refresh templates from GitHub."""
        async with self._lock:
            return await self._fetch_and_cache()

    def _ensure_fresh(self) -> bool:
        """Ensure the in-memory cache is fresh, else reload."""
        if not self._is_initialized:
            return False
        if self._last_memory_update and (datetime.now() - self._last_memory_update) > self._memory_ttl:
            # Trigger async reload, but for synchronous reads we just return current
            # The background loop will handle it if we use asyncio.create_task elsewhere.
            logger.debug("In-memory TTL expired, but returning cached data until background refresh")
        return True

    def get_template_for_style(self, style: str) -> Optional[str]:
        """
        Get a template for a specific style.
        Returns a single template string, or None if not found.
        """
        self._ensure_fresh()
        templates = self._templates_cache.get('styles', {}).get(style, [])
        if templates:
            return random.choice(templates)
        return None

    def get_templates_for_style(self, style: str) -> List[str]:
        """Get all templates for a specific style."""
        self._ensure_fresh()
        return self._templates_cache.get('styles', {}).get(style, [])

    def get_all_styles(self) -> List[str]:
        """Get all available style names."""
        self._ensure_fresh()
        return list(self._templates_cache.get('styles', {}).keys())

    def get_template(self, template_id: str) -> Optional[str]:
        """Get a specific template by ID."""
        self._ensure_fresh()
        return self._templates_cache.get('templates', {}).get(template_id)

    def get_all_templates(self) -> Dict[str, Any]:
        """Get all templates (for debugging)."""
        self._ensure_fresh()
        return self._templates_cache.copy()

    def get_quality_template(self) -> str:
        """Get a quality enhancement template."""
        self._ensure_fresh()
        templates = self._templates_cache.get('quality', [])
        if templates:
            return random.choice(templates)
        # Fallback local quality templates
        return self._local_templates.get_quality_template()

    def get_quality_templates_list(self) -> List[str]:
        """Get all quality enhancement templates."""
        self._ensure_fresh()
        templates = self._templates_cache.get('quality', [])
        if templates:
            return templates
        return self._local_templates._build_quality_templates()

    def get_composition_template(self, composition_type: str) -> Optional[str]:
        """Get a composition template by type."""
        self._ensure_fresh()
        templates = self._templates_cache.get('composition', {}).get(composition_type, [])
        if templates:
            return random.choice(templates)
        return None

    def get_composition_templates(self, composition_type: str) -> List[str]:
        """Get all composition templates for a type."""
        self._ensure_fresh()
        return self._templates_cache.get('composition', {}).get(composition_type, [])

    async def close(self):
        """Clean up resources."""
        await self._github_fetcher.close()

    async def get_info(self) -> Dict[str, Any]:
        """Return information about the template manager."""
        self._ensure_fresh()
        return {
            "type": "TemplateManager",
            "is_initialized": self._is_initialized,
            "total_templates": len(self._templates_cache.get('templates', {})),
            "total_styles": len(self._templates_cache.get('styles', {})),
            "last_update": self._last_update.isoformat() if self._last_update else None,
            "update_interval_minutes": self.update_interval.total_seconds() / 60,
            "cache_dir": str(self.cache_dir)
        }