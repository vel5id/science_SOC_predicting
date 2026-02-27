"""
File Utilities
==============
Helper functions for file existence checks and idempotency.
"""
from __future__ import annotations

from pathlib import Path


def should_skip_file(file_path: Path, min_size_bytes: int = 1) -> bool:
    """
    Check if file exists and has content.
    
    Args:
        file_path: Path to check
        min_size_bytes: Minimum file size in bytes (default: 1)
    
    Returns:
        True if file exists and size >= min_size_bytes, False otherwise
    """
    return file_path.exists() and file_path.stat().st_size >= min_size_bytes
