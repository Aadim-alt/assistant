# Performance Optimization Suggestions

import asyncio
import threading
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import weakref
from typing import Optional, Any
from dataclasses import dataclass
import time

# 1. Lazy Loading for AI Models
class LazyModelLoader:
    """Lazy loading for resource-intensive models"""
    
    def __init__(self):
        self._models = {}
        self._loading_locks = {}
    
    async def get_model(self, model_type: str, loader_func):
        """Get model with lazy loading"""
        if model_type not in self._models:
            # Prevent multiple simultaneous loads
            if model_type not in self._loading_locks:
                self._loading_locks[model_type] = asyncio.Lock()
            
            async with self._loading_locks[model_type]:
                if model_type not in self._models:
                    print(f"Loading {model_type} model...")
                    self._models[model_type] = await asyncio.to_thread(loader_func)
                    print(f"{model_type} model loaded")
        
        return self._models[model_type]

# 2. Connection Pooling for API Requests
import aiohttp
from aiohttp_retry import RetryClient, ExponentialRetry

class APIClient:
    """Optimized API client with connection pooling"""
    
    def __init__(self):
        self._session = None
        self._retry_client = None
    
    async def get_session(self):
        """Get or create aiohttp session with connection pooling"""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=100,  # Total connection pool size
                limit_per_host=30,  # Per-host connection limit
                ttl_dns_cache=300,  # DNS cache TTL
                use_dns_cache=True,
                enable_cleanup_closed=True
            )
            
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout
            )
            
            retry_options = ExponentialRetry(attempts=3, start_timeout=1)
            self._retry_client = RetryClient(
                client_session=self._session,
                retry_options=retry_options
            )
        
        return self._retry_client
    
    async def close(self):
        """Clean up session"""
        if self._session and not self._session.closed:
            await self._session.close()

# 3. Caching System
class ResponseCache:
    """LRU cache for responses with TTL"""
    
    def __init__(self, maxsize=1000, ttl=300):
        self.maxsize = maxsize
        self.ttl = ttl
        self._cache = {}
        self._timestamps = {}
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired"""
        if key in self._cache:
            if time.time() - self._timestamps[key] < self.ttl:
                return self._cache[key]
            else:
                # Expired, remove
                del self._cache[key]
                del self._timestamps[key]
        return None
    
    def set(self, key: str, value: Any):
        """Set cached value"""
        # Implement simple LRU eviction
        if len(self._cache) >= self.maxsize:
            oldest_key = min(self._timestamps.keys(), 
                           key=lambda k: self._timestamps[k])
            del self._cache[oldest_key]
            del self._timestamps[oldest_key]
        
        self._cache[key] = value
        self._timestamps[key] = time.time()

# 4. Optimized System Monitor
class OptimizedSystemMonitor:
    """System monitor with efficient data collection"""
    
    def __init__(self):
        self._last_cpu_times = None
        self._last_net_io = None
        self._cache = ResponseCache(maxsize=10, ttl=2)  # 2-second cache
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sysmon")
    
    async def get_system_info(self, force_refresh=False) -> dict:
        """Get system info with caching"""
        cache_key = "system_info"
        
        if not force_refresh:
            cached = self._cache.get(cache_key)
            if cached:
                return cached
        
        # Collect data in thread pool to avoid blocking
        info = await asyncio.get_event_loop().run_in_executor(
            self._executor, self._collect_system_data
        )
        
        self._cache.set(cache_key, info)
        return info
    
    def _collect_system_data(self) -> dict:
        """Collect system data efficiently"""
        import psutil
        
        # Use interval=None for non-blocking CPU percent
        cpu_percent = psutil.cpu_percent(interval=None)
        if self._last_cpu_times is None:
            # First call, get baseline
            psutil.cpu_percent(interval=0.1)
            cpu_percent = psutil.cpu_percent(interval=None)
        
        memory = psutil.virtual_memory()
        
        # Only get top 5 processes by CPU to reduce overhead
        processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent'], ad_value=0):
            try:
                if proc.info['cpu_percent'] > 1.0:
                    processes.append(proc.info)
                if len(processes) >= 5:
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return {
            "cpu_percent": cpu_percent,
            "memory": {
                "total": memory.total,
                "available": memory.available,
                "percent": memory.percent
            },
            "top_processes": processes,
            "timestamp": time.time()
        }

# 5. Background Task Manager
class BackgroundTaskManager:
    """Manage background tasks efficiently"""
    
    def __init__(self):
        self._tasks = weakref.WeakSet()
        self._shutdown_event = asyncio.Event()
    
    def create_task(self, coro, name=None):
        """Create and track background task"""
        task = asyncio.create_task(coro, name=name)
        self._tasks.add(task)
        
        # Add done callback to clean up
        task.add_done_callback(self._task_done_callback)
        return task
    
    def _task_done_callback(self, task):
        """Handle task completion"""
        if task.exception():
            logger.error(f"Background task {task.get_name()} failed: {task.exception()}")
    
    async def shutdown(self, timeout=30):
        """Gracefully shutdown all background tasks"""
        self._shutdown_event.set()
        
        if self._tasks:
            print(f"Shutting down {len(self._tasks)} background tasks...")
            await asyncio.wait_for(
                asyncio.gather(*self._tasks, return_exceptions=True),
                timeout=timeout
            )

# 6. Memory-Efficient Conversation History
from collections import deque
import pickle
import gzip

class ConversationHistory:
    """Memory-efficient conversation history management"""
    
    def __init__(self, max_memory_items=50, archive_path="conversation_archive.gz"):
        self._memory = deque(maxlen=max_memory_items)
        self._archive_path = archive_path
        self._archived_count = 0
    
    def add_message(self, role: str, content: str):
        """Add message to history"""
        message = {
            "role": role,
            "content": content,
            "timestamp": time.time()
        }
        
        # Archive old message if at capacity
        if len(self._memory) == self._memory.maxlen:
            self._archive_message(self._memory[0])
        
        self._memory.append(message)
    
    def _archive_message(self, message):
        """Archive message to compressed storage"""
        try:
            with gzip.open(self._archive_path, 'ab') as f:
                pickle.dump(message, f)
            self._archived_count += 1
        except Exception as e:
            logger.error(f"Failed to archive message: {e}")
    
    def get_recent_context(self, max_items=10) -> list:
        """Get recent conversation context"""
        return list(self._memory)[-max_items:]
    
    def get_stats(self) -> dict:
        """Get history statistics"""
        return {
            "memory_items": len(self._memory),
            "archived_items": self._archived_count,
            "total_items": len(self._memory) + self._archived_count
        }

# 7. Optimized GUI Updates
class GUIUpdateOptimizer:
    """Optimize GUI updates to prevent blocking"""
    
    def __init__(self, update_interval=100):  # milliseconds
        self._update_queue = asyncio.Queue()
        self._update_interval = update_interval
        self._last_update = 0
    
    async def schedule_update(self, widget, method, *args, **kwargs):
        """Schedule GUI update"""
        await self._update_queue.put((widget, method, args, kwargs))
    
    async def process_updates(self):
        """Process queued GUI updates"""
        current_time = time.time() * 1000
        
        if current_time - self._last_update < self._update_interval:
            return
        
        updates = []
        try:
            while True:
                update = self._update_queue.get_nowait()
                updates.append(update)
        except asyncio.QueueEmpty:
            pass
        
        if updates:
            # Batch process updates
            for widget, method, args, kwargs in updates:
                try:
                    method(*args, **kwargs)
                except Exception as e:
                    logger.error(f"GUI update failed: {e}")
            
            self._last_update = current_time

# Usage example in main JARVIS class:
class OptimizedJARVIS:
    def __init__(self):
        self.model_loader = LazyModelLoader()
        self.api_client = APIClient()
        self.response_cache = ResponseCache()
        self.task_manager = BackgroundTaskManager()
        self.conversation = ConversationHistory()
        self.gui_optimizer = GUIUpdateOptimizer()
    
    async def initialize(self):
        """Initialize components lazily"""
        # Only load models when needed
        pass
    
    async def shutdown(self):
        """Clean shutdown"""
        await self.task_manager.shutdown()
        await self.api_client.close()