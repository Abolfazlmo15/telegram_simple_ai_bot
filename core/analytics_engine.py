import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timedelta
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
        self.analytics_dir.mkdir(parents=True, exist_ok=True)

        self.running = False
        self.worker_thread = None
        self.interval_minutes = Config.ANALYTICS_INTERVAL_MINUTES
        self.interval_seconds = self.interval_minutes * 60

        # In-memory cache for quick access
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
                # Wait for interval
                for _ in range(self.interval_seconds):
                    if not self.running:
                        break
                    time.sleep(1)

                if not self.running:
                    break

                # Run analytics
                logger.info("Running analytics cycle...")
                self._run_analytics_cycle()

            except Exception as e:
                logger.error(f"Analytics worker error: {e}", exc_info=True)

        logger.info("Analytics worker loop ended")

    def _run_analytics_cycle(self):
        """Execute one analytics cycle"""
        start_time = time.time()

        # Analyze all users
        user_dirs = [d for d in self.users_dir.iterdir() if d.is_dir() and d.name != "analytics"]

        for user_dir in user_dirs:
            try:
                user_id = int(user_dir.name)
                self._analyze_user(user_id, user_dir)
            except Exception as e:
                logger.error(f"Error analyzing user {user_dir.name}: {e}")

        # Update global stats
        self._update_global_stats()

        # Save analytics
        self._save_analytics()

        duration = time.time() - start_time
        logger.info(f"Analytics cycle completed in {duration:.2f}s")

    def _analyze_user(self, user_id: int, user_dir: Path):
        """Analyze a single user's data"""
        # Load user data
        user_file = user_dir / "user_data.json"
        if not user_file.exists():
            return

        with open(user_file, 'r', encoding='utf-8') as f:
            user_data = json.load(f)

        # Extract metrics
        history = user_data.get('history', [])
        stats = user_data.get('stats', {})

        # Analyze conversation patterns
        categories = defaultdict(int)
        response_times = []
        topics = defaultdict(int)
        active_hours = defaultdict(int)

        for entry in history:
            # Extract category if present
            if 'category' in entry:
                categories[entry['category']] += 1

            # Extract timestamp and hour
            if 'timestamp' in entry:
                try:
                    dt = datetime.fromisoformat(entry['timestamp'])
                    active_hours[dt.hour] += 1
                except:
                    pass

            # Extract topics from messages
            if 'text' in entry:
                text = entry['text'].lower()
                # Simple topic extraction (can be enhanced with NLP)
                if 'code' in text or 'function' in text:
                    topics['programming'] += 1
                elif 'explain' in text or 'what is' in text:
                    topics['education'] += 1
                elif 'business' in text or 'strategy' in text:
                    topics['business'] += 1

        # Calculate metrics
        total_messages = len(history)
        preferred_category = max(categories.items(), key=lambda x: x[1])[0] if categories else "unknown"
        most_active_hours = sorted(active_hours.items(), key=lambda x: x[1], reverse=True)[:3]
        most_active_hours = [h[0] for h in most_active_hours]

        # Create metrics object
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

        # Cache metrics
        self.user_metrics_cache[user_id] = metrics

        # Create analytics summary
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

        # Save user analytics
        analytics_file = self.analytics_dir / f"{user_id}.json"
        with open(analytics_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(analytics), f, indent=2)

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
        with open(global_file, 'w', encoding='utf-8') as f:
            json.dump(self.global_stats, f, indent=2)

    def get_user_analytics(self, user_id: int) -> Optional[UserAnalytics]:
        """Get analytics for a specific user"""
        analytics_file = self.analytics_dir / f"{user_id}.json"
        if not analytics_file.exists():
            return None

        with open(analytics_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return UserAnalytics(**data)

    def get_global_stats(self) -> Dict:
        """Get global statistics"""
        return self.global_stats.copy()

    def record_message(self, user_id: int, category: str, response_time: float,
                       tokens_used: int = 0):
        """
        Record a message for later analysis.
        This is called by the bot handlers for each message.
        """
        # Update in-memory metrics
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
            # Update average response time
            metrics.average_response_time = (
                    (metrics.average_response_time * (metrics.total_messages - 1) + response_time)
                    / metrics.total_messages
            )
            metrics.last_active = datetime.now().isoformat()