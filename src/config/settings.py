"""Configuration management for the Rabi MCP server."""

import os
from typing import Any, Dict, Optional
from pydantic import BaseSettings, Field, validator
from enum import Enum


class ComputationalBackend(str, Enum):
    """Available computational backends."""
    NUMPY = "numpy"
    JAX = "jax"
    NUMBA = "numba"


class Precision(str, Enum):
    """Numerical precision options."""
    SINGLE = "single"
    DOUBLE = "double"


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Server configuration settings."""
    
    # Server Settings
    server_name: str = Field(default="rabi-mcp-server", env="MCP_SERVER_NAME")
    server_version: str = Field(default="1.0.0", env="MCP_SERVER_VERSION")
    port: int = Field(default=8000, env="PORT")
    host: str = Field(default="0.0.0.0", env="HOST")
    
    # Computational Settings
    computational_backend: ComputationalBackend = Field(
        default=ComputationalBackend.NUMPY, 
        env="COMPUTATIONAL_BACKEND"
    )
    max_hilbert_dim: int = Field(default=1000, env="MAX_HILBERT_DIM")
    enable_gpu: bool = Field(default=False, env="ENABLE_GPU")
    precision: Precision = Field(default=Precision.DOUBLE, env="PRECISION")
    enable_parallel: bool = Field(default=True, env="ENABLE_PARALLEL")
    cache_results: bool = Field(default=True, env="CACHE_RESULTS")
    
    # Performance Settings
    num_threads: int = Field(default=4, env="NUM_THREADS")
    memory_limit_gb: int = Field(default=8, env="MEMORY_LIMIT_GB")
    enable_jit_compilation: bool = Field(default=True, env="ENABLE_JIT_COMPILATION")
    
    # Logging Configuration
    log_level: LogLevel = Field(default=LogLevel.INFO, env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    enable_performance_logging: bool = Field(default=True, env="ENABLE_PERFORMANCE_LOGGING")
    
    # Optional: Redis Configuration
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")
    
    # Optional: GPU Configuration
    cuda_visible_devices: Optional[str] = Field(default=None, env="CUDA_VISIBLE_DEVICES")
    gpu_memory_fraction: float = Field(default=0.8, env="GPU_MEMORY_FRACTION")
    
    # Security Settings
    api_key: Optional[str] = Field(default=None, env="API_KEY")
    allowed_origins: str = Field(default="*", env="ALLOWED_ORIGINS")
    enable_cors: bool = Field(default=True, env="ENABLE_CORS")
    
    # Development Settings
    debug: bool = Field(default=False, env="DEBUG")
    reload: bool = Field(default=False, env="RELOAD")
    workers: int = Field(default=1, env="WORKERS")
    
    @validator("max_hilbert_dim")
    def validate_hilbert_dim(cls, v):
        """Validate Hilbert space dimension."""
        if v < 2:
            raise ValueError("Hilbert space dimension must be at least 2")
        if v > 50000:
            raise ValueError("Hilbert space dimension too large (max 50000)")
        return v
    
    @validator("num_threads")
    def validate_threads(cls, v):
        """Validate number of threads."""
        if v < 1:
            raise ValueError("Number of threads must be at least 1")
        return v
    
    @validator("memory_limit_gb")
    def validate_memory(cls, v):
        """Validate memory limit."""
        if v < 1:
            raise ValueError("Memory limit must be at least 1 GB")
        return v
    
    @validator("gpu_memory_fraction")
    def validate_gpu_memory(cls, v):
        """Validate GPU memory fraction."""
        if not 0.1 <= v <= 1.0:
            raise ValueError("GPU memory fraction must be between 0.1 and 1.0")
        return v
    
    class Config:
        """Pydantic config."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        use_enum_values = True
    
    def get_numpy_config(self) -> Dict[str, Any]:
        """Get NumPy configuration."""
        return {
            "dtype": "float64" if self.precision == Precision.DOUBLE else "float32",
            "threads": self.num_threads if self.enable_parallel else 1,
        }
    
    def get_computation_config(self) -> Dict[str, Any]:
        """Get computation configuration."""
        return {
            "backend": self.computational_backend,
            "max_hilbert_dim": self.max_hilbert_dim,
            "enable_gpu": self.enable_gpu,
            "precision": self.precision,
            "enable_parallel": self.enable_parallel,
            "cache_results": self.cache_results,
            "num_threads": self.num_threads,
            "enable_jit": self.enable_jit_compilation,
        }
    
    def setup_environment(self):
        """Set up environment variables based on configuration."""
        # Set NumPy threading
        if not self.enable_parallel:
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["NUMEXPR_NUM_THREADS"] = "1"
        else:
            os.environ["OMP_NUM_THREADS"] = str(self.num_threads)
            os.environ["MKL_NUM_THREADS"] = str(self.num_threads)
            os.environ["NUMEXPR_NUM_THREADS"] = str(self.num_threads)
        
        # Set GPU configuration
        if self.cuda_visible_devices:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        
        # Set precision
        if self.precision == Precision.SINGLE:
            os.environ["NUMPY_EXPERIMENTAL_DTYPE_API"] = "1"


# Global settings instance
settings = Settings()