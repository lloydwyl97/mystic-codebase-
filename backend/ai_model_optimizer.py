"""
AI Model Optimizer for Mystic Trading Platform

Provides optimized AI model loading with:
- Lazy loading
- Model caching
- Memory management
- Performance monitoring
- Background loading
"""

import asyncio
import gc
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import threading
from concurrent.futures import ThreadPoolExecutor

try:
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModel, pipeline
    from sentence_transformers import SentenceTransformer
    import joblib
except ImportError:
    torch = None
    np = None
    AutoTokenizer = None
    AutoModel = None
    pipeline = None
    SentenceTransformer = None
    joblib = None


logger = logging.getLogger(__name__)

# Model configuration
MODEL_CACHE_DIR = "models/cache"
MODEL_DOWNLOAD_TIMEOUT = 300  # 5 minutes
MODEL_LOAD_TIMEOUT = 60  # 1 minute
MAX_MODEL_MEMORY = 0.8  # 80% of available memory
BACKGROUND_LOADING = True


@dataclass
class ModelInfo:
    """Model information and metadata"""
    name: str
    type: str  # 'transformer', 'sentiment', 'embedding', 'custom'
    path: str
    size_mb: float
    loaded: bool = False
    load_time: float = 0.0
    memory_usage: float = 0.0
    last_used: float = 0.0
    access_count: int = 0


class ModelCache:
    """In-memory model cache with LRU eviction"""

    def __init__(self, max_size: int = 10):
        self.max_size = max_size
        self.models: Dict[str, Any] = {}
        self.model_info: Dict[str, ModelInfo] = {}
        self.access_order: List[str] = []
        self.lock = threading.Lock()

    def get(self, model_name: str) -> Optional[Any]:
        """Get model from cache"""
        with self.lock:
            if model_name in self.models:
                # Update access order
                if model_name in self.access_order:
                    self.access_order.remove(model_name)
                self.access_order.append(model_name)

                # Update model info
                if model_name in self.model_info:
                    self.model_info[model_name].access_count += 1
                    self.model_info[model_name].last_used = time.time()

                return self.models[model_name]
            return None

    def set(self, model_name: str, model: Any, model_info: ModelInfo):
        """Add model to cache"""
        with self.lock:
            # Evict least recently used if cache is full
            if len(self.models) >= self.max_size and model_name not in self.models:
                self._evict_lru()

            self.models[model_name] = model
            self.model_info[model_name] = model_info

            # Update access order
            if model_name in self.access_order:
                self.access_order.remove(model_name)
            self.access_order.append(model_name)

    def _evict_lru(self):
        """Evict least recently used model"""
        if self.access_order:
            lru_model = self.access_order[0]
            del self.models[lru_model]
            del self.model_info[lru_model]
            self.access_order.pop(0)
            logger.info(f"Evicted model from cache: {lru_model}")

    def clear(self):
        """Clear all models from cache"""
        with self.lock:
            self.models.clear()
            self.model_info.clear()
            self.access_order.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            return {
                'cache_size': len(self.models),
                'max_size': self.max_size,
                'models': list(self.models.keys()),
                'total_memory_mb': sum(info.memory_usage for info in self.model_info.values()),
                'access_order': self.access_order.copy()
            }


class ModelLoader:
    """Optimized model loader with background loading"""

    def __init__(self):
        self.cache = ModelCache()
        self.loading_models: Dict[str, asyncio.Future] = {}
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.lock = threading.Lock()

        # Ensure cache directory exists
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    async def load_model(self, model_name: str, model_type: str = "transformer") -> Any:
        """Load model with caching and background loading"""
        # Check cache first
        cached_model = self.cache.get(model_name)
        if cached_model is not None:
            logger.debug(f"Model {model_name} found in cache")
            return cached_model

        # Check if model is already loading
        if model_name in self.loading_models:
            logger.debug(f"Model {model_name} is already loading")
            return await self.loading_models[model_name]

        # Start loading model
        logger.info(f"Loading model: {model_name}")
        future = asyncio.create_task(self._load_model_async(model_name, model_type))
        self.loading_models[model_name] = future

        try:
            model = await future
            return model
        finally:
            # Clean up loading state
            self.loading_models.pop(model_name, None)

    async def _load_model_async(self, model_name: str, model_type: str) -> Any:
        """Load model asynchronously"""
        try:
            # Run model loading in thread pool
            loop = asyncio.get_event_loop()
            model, model_info = await loop.run_in_executor(
                self.executor,
                self._load_model_sync,
                model_name,
                model_type
            )

            # Add to cache
            self.cache.set(model_name, model, model_info)

            logger.info(f"Model {model_name} loaded successfully")
            return model

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def _load_model_sync(self, model_name: str, model_type: str) -> tuple[Any, ModelInfo]:
        """Load model synchronously in thread"""
        start_time = time.time()

        try:
            if model_type == "sentiment":
                model = self._load_sentiment_model(model_name)
            elif model_type == "embedding":
                model = self._load_embedding_model(model_name)
            elif model_type == "transformer":
                model = self._load_transformer_model(model_name)
            else:
                model = self._load_custom_model(model_name)

            load_time = time.time() - start_time
            memory_usage = self._estimate_memory_usage(model)

            model_info = ModelInfo(
                name=model_name,
                type=model_type,
                path=model_name,
                size_mb=memory_usage,
                loaded=True,
                load_time=load_time,
                memory_usage=memory_usage,
                last_used=time.time(),
                access_count=1
            )

            return model, model_info

        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise

    def _load_sentiment_model(self, model_name: str) -> Any:
        """Load sentiment analysis model"""
        if pipeline is None:
            raise ImportError("transformers library not available")

        return pipeline(
            "sentiment-analysis",
            model=model_name,
            device=0 if torch and torch.cuda.is_available() else -1
        )

    def _load_embedding_model(self, model_name: str) -> Any:
        """Load embedding model"""
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers library not available")

        return SentenceTransformer(model_name)

    def _load_transformer_model(self, model_name: str) -> Any:
        """Load transformer model"""
        if AutoTokenizer is None or AutoModel is None:
            raise ImportError("transformers library not available")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        return {"tokenizer": tokenizer, "model": model}

    def _load_custom_model(self, model_name: str) -> Any:
        """Load custom model from file"""
        if joblib is None:
            raise ImportError("joblib library not available")

        model_path = os.path.join(MODEL_CACHE_DIR, f"{model_name}.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        return joblib.load(model_path)

    def _estimate_memory_usage(self, model: Any) -> float:
        """Estimate model memory usage in MB"""
        try:
            if torch and hasattr(model, 'parameters'):
                # Count parameters for PyTorch models
                total_params = sum(p.numel() for p in model.parameters())
                # Estimate 4 bytes per parameter (float32)
                memory_bytes = total_params * 4
                return memory_bytes / (1024 * 1024)  # Convert to MB
            else:
                # Rough estimate for other models
                return 100.0  # Default 100MB estimate
        except Exception:
            return 100.0  # Default estimate

    async def preload_models(self, model_configs: List[Dict[str, str]]):
        """Preload models in background"""
        if not BACKGROUND_LOADING:
            return

        logger.info(f"Preloading {len(model_configs)} models")

        # Start background loading
        tasks = []
        for config in model_configs:
            model_name = config['name']
            model_type = config.get('type', 'transformer')

            # Skip if already loaded or loading
            if (model_name in self.cache.models or
                model_name in self.loading_models):
                continue

            task = asyncio.create_task(
                self.load_model(model_name, model_type)
            )
            tasks.append(task)

        # Wait for all models to load
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("Model preloading completed")

    def unload_model(self, model_name: str):
        """Unload model from cache"""
        with self.lock:
            if model_name in self.cache.models:
                del self.cache.models[model_name]
                del self.cache.model_info[model_name]
                if model_name in self.cache.access_order:
                    self.cache.access_order.remove(model_name)

                # Force garbage collection
                gc.collect()

                logger.info(f"Unloaded model: {model_name}")

    def optimize_memory(self):
        """Optimize memory usage by unloading least used models"""
        with self.lock:
            if len(self.cache.models) <= 1:
                return

            # Calculate total memory usage
            total_memory = sum(info.memory_usage for info in self.cache.model_info.values())

            # If memory usage is high, unload least used models
            if total_memory > 1000:  # 1GB threshold
                # Sort by last used time
                sorted_models = sorted(
                    self.cache.model_info.items(),
                    key=lambda x: x[1].last_used
                )

                # Unload oldest 50% of models
                unload_count = len(sorted_models) // 2
                for model_name, _ in sorted_models[:unload_count]:
                    self.unload_model(model_name)

                logger.info(f"Memory optimization: unloaded {unload_count} models")

    def get_model_stats(self) -> Dict[str, Any]:
        """Get model loading statistics"""
        with self.lock:
            stats = self.cache.get_stats()
            stats.update({
                'loading_models': list(self.loading_models.keys()),
                'total_models': len(self.cache.models) + len(self.loading_models),
                'available_memory_mb': self._get_available_memory()
            })
            return stats

    def _get_available_memory(self) -> float:
        """Get available system memory in MB"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.available / (1024 * 1024)  # Convert to MB
        except ImportError:
            return 1000.0  # Default estimate

    def cleanup(self):
        """Cleanup resources"""
        self.cache.clear()
        self.executor.shutdown(wait=True)
        logger.info("Model loader cleaned up")


# Global model loader instance
model_loader = ModelLoader()


