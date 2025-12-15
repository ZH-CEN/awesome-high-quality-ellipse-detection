"""
High-quality Ellipse Detection - Pure Python Implementation

This module provides a Python implementation of the arc-support line segment based
ellipse detection algorithm from the paper:
"Arc-support Line Segments Revisited: An Efficient and High-quality Ellipse Detection"

Author: Refactored from MATLAB/C++ implementation
"""

from .ellipse_detector import EllipseDetector

__all__ = ['EllipseDetector']
__version__ = '1.0.0'
