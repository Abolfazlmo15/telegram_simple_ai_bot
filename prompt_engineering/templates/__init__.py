"""Template management for prompt engineering."""
from prompt_engineering.templates.template_manager import TemplateManager
from prompt_engineering.templates.github_fetcher import GithubFetcher
from prompt_engineering.templates.local_templates import LocalTemplates

__all__ = [
    'TemplateManager',
    'GithubFetcher',
    'LocalTemplates'
]