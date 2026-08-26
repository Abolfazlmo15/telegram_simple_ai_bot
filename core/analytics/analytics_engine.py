import asyncio
import json
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict
from core.config import Config

logger = logging.getLogger(__name__)


@dataclass
class ConversationMetrics:
    """Metrics for a single conversation"""
    user_id: int
    total_messages: int
    total_tokens_used: int
    average_response_time: float
    preferred_category: str
    last_active: str
    session_start: str
    common_topics: List[str]


@dataclass
class UserAnalytics:
    """Analytics data for a user"""
    user_id: int
    total_conversations: int
    total_messages: int
    favorite_categories: Dict[str, int]
    average_session_duration: float
    most_active_hours: List[int]
    response_quality_score: float
    last_analysis: str


class AnalyticsEngine:
    """
    Background analytics processor that runs independently of the main bot.
    Analyzes user behavior, conversation patterns, and system performance.
    Runs every N minutes (configurable) without blocking bot operations.
    """

    def __init__(self, users_dir: str = Config.USER_DATA_DIR):
        self.users_dir = Path(users_dir)
        self.analytics_dir = self.users_dir / "analytics"

        # Ensure directory exists on initialization
        self._ensure_dir_exists()

        self.running = False
        self.worker_thread = None
        self.interval_minutes = Config.ANALYTICS_INTERVAL_MINUTES
        self.interval_seconds = self.interval_minutes * 60

        self.user_metrics_cache: Dict[int, ConversationMetrics] = {}
        self.global_stats = {
            "total_users": 0,
            "total_messages": 0,
            "total_api_calls": 0,
            "average_response_time": 0.0,
            "error_rate": 0.0,
            "last_updated": None
        }

        logger.info(f"Analytics engine initialized (interval: {self.interval_minutes} min)")

    def _ensure_dir_exists(self):
        """Ensure the analytics directory exists."""
        try:
            self.analytics_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create analytics directory: {e}")

    def start(self):
        """Start the background analytics worker"""
        if self.running:
            logger.warning("Analytics engine already running")
            return
        self.running = True
        self.worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="AnalyticsWorker"
        )
        self.worker_thread.start()
        logger.info("Analytics engine started")

    def stop(self):
        """Stop the background analytics worker"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5.0)
        logger.info("Analytics engine stopped")

    def _worker_loop(self):
        """Main worker loop - runs periodically"""
        logger.info("Analytics worker loop started")
        while self.running:
            try:
                for _ in range(self.interval_seconds):
                    if not self.running:
                        break
                    time.sleep(1)
                if not self.running:
                    break

                logger.info("Running analytics cycle...")
                self._run_analytics_cycle()
            except Exception as e:
                logger.error(f"Analytics worker error: {e}", exc_info=True)
                # Prevent infinite crash loop by sleeping a bit
                time.sleep(10)
        logger.info("Analytics worker loop ended")

    def _run_analytics_cycle(self):
        """Execute one analytics cycle"""
        start_time = time.time()

        # Ensure directory exists before starting cycle
        self._ensure_dir_exists()

        # Only analyze numeric directory names (actual user IDs)
        user_dirs = [
            d for d in self.users_dir.iterdir()
            if d.is_dir() and d.name != "analytics" and d.name.isdigit()
        ]

        for user_dir in user_dirs:
            try:
                user_id = int(user_dir.name)
                self._analyze_user(user_id, user_dir)
            except Exception as e:
                logger.error(f"Error analyzing user {user_dir.name}: {e}")

        self._update_global_stats()
        self._save_analytics()

        # Generate and save global report
        report = self.generate_global_report()
        report_file = self.analytics_dir / "global_report.json"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            logger.info("Global report saved.")
        except Exception as e:
            logger.error(f"Failed to save global report: {e}")

        duration = time.time() - start_time
        logger.info(f"Analytics cycle completed in {duration:.2f}s")

    def _analyze_user(self, user_id: int, user_dir: Path):
        """Analyze a single user's data"""
        user_file = user_dir / "user_data.json"
        if not user_file.exists():
            return

        try:
            with open(user_file, 'r', encoding='utf-8') as f:
                user_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read user data for {user_id}: {e}")
            return

        history = user_data.get('history', [])
        stats = user_data.get('stats', {})

        categories = defaultdict(int)
        topics = defaultdict(int)
        active_hours = defaultdict(int)

        for entry in history:
            if 'category' in entry:
                categories[entry['category']] += 1

            if 'timestamp' in entry:
                try:
                    dt = datetime.fromisoformat(entry['timestamp'])
                    active_hours[dt.hour] += 1
                except Exception:
                    pass

            if 'text' in entry:
                text = entry['text'].lower()
                if 'code' in text or 'function' in text:
                    topics['programming'] += 1
                elif 'explain' in text or 'what is' in text:
                    topics['education'] += 1
                elif 'business' in text or 'strategy' in text:
                    topics['business'] += 1

        total_messages = len(history)
        preferred_category = max(categories.items(), key=lambda x: x[1])[0] if categories else "unknown"
        most_active_hours = sorted(active_hours.items(), key=lambda x: x[1], reverse=True)[:3]
        most_active_hours = [h[0] for h in most_active_hours]

        metrics = ConversationMetrics(
            user_id=user_id,
            total_messages=total_messages,
            total_tokens_used=stats.get('total_tokens', 0),
            average_response_time=stats.get('avg_response_time', 0.0),
            preferred_category=preferred_category,
            last_active=datetime.now().isoformat(),
            session_start=user_data.get('created_at', datetime.now().isoformat()),
            common_topics=list(topics.keys())[:5]
        )

        self.user_metrics_cache[user_id] = metrics

        analytics = UserAnalytics(
            user_id=user_id,
            total_conversations=stats.get('total_conversations', 1),
            total_messages=total_messages,
            favorite_categories=dict(categories),
            average_session_duration=stats.get('avg_session_duration', 0.0),
            most_active_hours=most_active_hours,
            response_quality_score=stats.get('quality_score', 0.0),
            last_analysis=datetime.now().isoformat()
        )

        analytics_file = self.analytics_dir / f"{user_id}.json"
        try:
            # Ensure directory exists right before writing
            self.analytics_dir.mkdir(parents=True, exist_ok=True)
            with open(analytics_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(analytics), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save analytics for user {user_id}: {e}")

    def _update_global_stats(self):
        """Update global statistics"""
        total_users = len(self.user_metrics_cache)
        total_messages = sum(m.total_messages for m in self.user_metrics_cache.values())
        self.global_stats.update({
            "total_users": total_users,
            "total_messages": total_messages,
            "last_updated": datetime.now().isoformat()
        })

    def _save_analytics(self):
        """Save global analytics"""
        global_file = self.analytics_dir / "global_stats.json"
        try:
            # Ensure directory exists right before writing
            self.analytics_dir.mkdir(parents=True, exist_ok=True)
            with open(global_file, 'w', encoding='utf-8') as f:
                json.dump(self.global_stats, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save global analytics: {e}")

    def get_user_analytics(self, user_id: int) -> Optional[UserAnalytics]:
        """Get analytics for a specific user"""
        analytics_file = self.analytics_dir / f"{user_id}.json"
        if not analytics_file.exists():
            return None
        try:
            with open(analytics_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return UserAnalytics(**data)
        except Exception as e:
            logger.error(f"Failed to read analytics for user {user_id}: {e}")
            return None

    def get_global_stats(self) -> Dict:
        """Get global statistics"""
        return self.global_stats.copy()

    def record_message(self, user_id: int, category: str, response_time: float, tokens_used: int = 0):
        """Record a message for later analysis."""
        try:
            if user_id not in self.user_metrics_cache:
                self.user_metrics_cache[user_id] = ConversationMetrics(
                    user_id=user_id,
                    total_messages=1,
                    total_tokens_used=tokens_used,
                    average_response_time=response_time,
                    preferred_category=category,
                    last_active=datetime.now().isoformat(),
                    session_start=datetime.now().isoformat(),
                    common_topics=[]
                )
            else:
                metrics = self.user_metrics_cache[user_id]
                metrics.total_messages += 1
                metrics.total_tokens_used += tokens_used
                metrics.average_response_time = (
                        (metrics.average_response_time * (metrics.total_messages - 1) + response_time)
                        / metrics.total_messages
                )
                metrics.last_active = datetime.now().isoformat()
        except Exception as e:
            logger.error(f"Failed to record message for user {user_id}: {e}")

    # ---------- NEW ANALYTICS METHODS ----------
    def _extract_topics(self, history: List[Dict]) -> Dict[str, int]:
        """Extract topics from message history."""
        topics = defaultdict(int)
        question_words = ["what", "how", "why", "when", "where", "who", "which", "can", "does", "is", "are"]
        for entry in history:
            msg = entry.get('message', '').lower()
            # Count questions
            if any(msg.startswith(w) for w in question_words):
                topics["questions"] += 1
            # Detect categories from prompt_library
            if 'category' in entry:
                topics[entry['category']] += 1
            # Keyword extraction
            for kw in ["code", "python", "javascript", "function", "error", "bug", "fix", "explain", "teach"]:
                if kw in msg:
                    topics[kw] += 1
        return dict(topics)

    def _generate_user_report(self, user_id: int) -> Dict:
        """Generate a report for a single user."""
        user_file = self.users_dir / str(user_id) / "user_data.json"
        if not user_file.exists():
            return {}
        try:
            with open(user_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            return {}
        history = data.get('history', [])
        topics = self._extract_topics(history)
        total = len(history)
        avg_response = data['stats'].get('avg_response_time', 0)
        return {
            "total_messages": total,
            "top_topics": sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5],
            "avg_response_time": avg_response,
            "total_sessions": data['stats'].get('total_conversations', 0)
        }

    def generate_global_report(self) -> Dict:
        """Generate a global analytics report."""
        total_users = 0
        total_messages = 0
        total_topics = defaultdict(int)
        all_response_times = []
        for user_dir in self.users_dir.iterdir():
            if not user_dir.is_dir() or not user_dir.name.isdigit():
                continue
            report = self._generate_user_report(int(user_dir.name))
            if report:
                total_users += 1
                total_messages += report['total_messages']
                all_response_times.append(report['avg_response_time'])
                for topic, count in report['top_topics']:
                    total_topics[topic] += count
        return {
            "total_users": total_users,
            "total_messages": total_messages,
            "avg_response_time": sum(all_response_times) / len(all_response_times) if all_response_times else 0,
            "global_top_topics": sorted(total_topics.items(), key=lambda x: x[1], reverse=True)[:10]
        }

    def clear_cache(self) -> None:
        """
        Clear in-memory metrics cache and reset global stats.
        Does not affect stored analytics files on disk.
        """
        self.user_metrics_cache.clear()
        self.global_stats = {
            "total_users": 0,
            "total_messages": 0,
            "total_api_calls": 0,
            "average_response_time": 0.0,
            "error_rate": 0.0,
            "last_updated": None
        }
        logger.info("🧹 AnalyticsEngine cache cleared (metrics reset)")

    def get_info(self) -> Dict[str, any]:
        """Return information about the analytics engine."""
        return {
            "type": "AnalyticsEngine",
            "running": self.running,
            "interval_minutes": self.interval_minutes,
            "cached_users": len(self.user_metrics_cache),
            "analytics_dir": str(self.analytics_dir)
        }