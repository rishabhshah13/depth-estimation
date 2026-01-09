"""
Depth Estimation Comparison Package

This package provides interfaces for three depth estimation models:
- MiDaS: Robust monocular depth estimation
- Depth Anything V2: Foundation model for depth estimation
- Marigold: Diffusion-based depth estimation

Example usage:
    from src.midas_depth import MiDaSDepthEstimator
    from src.depth_anything import DepthEstimator as DepthAnythingEstimator
    from src.marigold_depth import MarigoldDepthEstimator

    # Initialize models
    midas = MiDaSDepthEstimator()
    depth_anything = DepthAnythingEstimator()
    marigold = MarigoldDepthEstimator()

    # Estimate depth
    depth_map = midas.estimate_depth("image.jpg")
"""

__version__ = "1.0.0"
__all__ = [
    "MiDaSDepthEstimator",
    "DepthEstimator",
    "MarigoldDepthEstimator",
    "DepthComparisonApp",
    "DepthEstimationGUI"
]
