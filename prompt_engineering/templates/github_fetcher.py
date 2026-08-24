"""
Fetches prompt templates from GitHub repositories.
Supports various prompt engineering repositories.
"""
import logging
import json
import httpx
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class GithubFetcher:
    """
    Fetches prompt templates from GitHub repositories.

    Repositories to fetch:
    - Prompt engineering examples
    - Stable Diffusion prompt templates
    - Midjourney style prompts
    - Custom prompt collections
    """

    def __init__(self):
        self.github_raw_base = "https://raw.githubusercontent.com"
        self.repositories = [
            # Prompt engineering repositories
            {
                "owner": "prompt-engineering",
                "repo": "prompt-engineering",
                "path": "templates",
                "file": "prompts.json"
            },
            # Stable Diffusion prompt examples
            {
                "owner": "stability-ai",
                "repo": "prompt-library",
                "path": "",
                "file": "prompts.json"
            },
            # Custom fallback: we can also fetch from a specific gist or repo
            # For now, we'll use a list of known good templates directly
        ]
        self._client = httpx.AsyncClient(timeout=15.0)
        logger.info("🌐 GithubFetcher initialized")

    async def fetch_templates(self) -> Dict[str, Any]:
        """
        Fetch templates from GitHub repositories.
        Returns a dictionary with templates categorized by style and type.
        """
        all_templates = {
            "styles": {},
            "templates": {},
            "quality": [],
            "composition": {},
            "last_fetched": datetime.now().isoformat()
        }

        for repo in self.repositories:
            try:
                url = self._build_url(repo)
                logger.info(f"🌐 Fetching from: {url}")

                resp = await self._client.get(url)
                resp.raise_for_status()

                data = resp.json()

                # Merge the fetched data
                all_templates = self._merge_templates(all_templates, data)

            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    logger.warning(f"Repo not found: {repo['repo']}")
                else:
                    logger.error(f"HTTP error fetching {repo['repo']}: {e}")
            except Exception as e:
                logger.error(f"Failed to fetch from {repo['repo']}: {e}")
                continue

        # If no templates were fetched, use built-in fallback
        if not all_templates.get('styles') and not all_templates.get('templates'):
            logger.warning("No templates fetched, using built-in fallback")
            from prompt_engineering.templates.local_templates import LocalTemplates
            local = LocalTemplates()
            all_templates = local.get_all_templates()
            all_templates['last_fetched'] = datetime.now().isoformat()

        return all_templates

    def _build_url(self, repo: Dict) -> str:
        """Build the raw GitHub URL for the template file."""
        owner = repo['owner']
        repo_name = repo['repo']
        path = repo.get('path', '')
        file = repo.get('file', 'prompts.json')

        if path:
            full_path = f"{path}/{file}".strip('/')
        else:
            full_path = file

        return f"{self.github_raw_base}/{owner}/{repo_name}/main/{full_path}"

    def _merge_templates(self, existing: Dict, new: Dict) -> Dict:
        """Merge new templates into existing."""
        # Styles
        if 'styles' in new:
            for style, templates in new['styles'].items():
                if style not in existing['styles']:
                    existing['styles'][style] = []
                existing['styles'][style].extend(templates)

        # Templates
        if 'templates' in new:
            for tid, template in new['templates'].items():
                if tid not in existing['templates']:
                    existing['templates'][tid] = template

        # Quality
        if 'quality' in new:
            existing['quality'].extend(new['quality'])

        # Composition
        if 'composition' in new:
            for comp, templates in new['composition'].items():
                if comp not in existing['composition']:
                    existing['composition'][comp] = []
                existing['composition'][comp].extend(templates)

        return existing

    async def close(self):
        """Close the HTTP client."""
        await self._client.aclose()

    def get_info(self) -> Dict[str, Any]:
        """Return information about the fetcher."""
        return {
            "type": "GithubFetcher",
            "repositories": len(self.repositories),
            "base_url": self.github_raw_base
        }