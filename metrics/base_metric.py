"""
Abstract base class for all metrics
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path


class BaseMetric(ABC):
    """
    Abstract base class for image evaluation metrics
    """
    
    def __init__(self, model: Optional[Any] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize metric with model and configuration
        
        Args:
            model: Model instance (from models package) for feature extraction/scoring
            config: Configuration dictionary for the metric
        """
        self.model = model
        self.config = config or {}
        self.results = {}
        
    @abstractmethod
    def compute(self, real_path: Path, fake_path: Path, **kwargs) -> Dict[str, Any]:
        """
        Compute the metric
        
        Args:
            real_path: Path to real images directory
            fake_path: Path to fake/generated images directory
            **kwargs: Additional arguments specific to each metric
            
        Returns:
            Dictionary containing metric results
        """
        pass
    
    @abstractmethod
    def interpret_result(self, result: Dict[str, Any]) -> str:
        """
        Interpret the metric result and provide human-readable explanation
        
        Args:
            result: Result dictionary from compute()
            
        Returns:
            Human-readable interpretation string
        """
        pass
    
    def save_results(self, output_path: Path) -> None:
        """
        Save results to file
        
        Args:
            output_path: Path to save results
        """
        import json
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
    
    def get_metric_name(self) -> str:
        """
        Get the name of the metric
        
        Returns:
            Metric name
        """
        return self.__class__.__name__
    
    def get_image_paths(self, path_or_paths) -> list:
        """
        Get image paths from directory, single file, or list of files
        
        Args:
            path_or_paths: Can be:
                - Path/str to a directory (will find all images inside)
                - Path/str to a single image file
                - List of Path/str to multiple image files
                
        Returns:
            Sorted list of image file paths (as strings)
        """
        import glob
        from pathlib import Path
        
        # Handle list of paths
        if isinstance(path_or_paths, (list, tuple)):
            image_paths = []
            for p in path_or_paths:
                p = Path(p)
                if p.is_file() and self._is_image_file(p):
                    image_paths.append(str(p))
                elif p.is_dir():
                    image_paths.extend(self._get_images_from_directory(p))
            return sorted(image_paths)
        
        # Handle single path
        path = Path(path_or_paths)
        
        # If it's a file, return it directly (if it's an image)
        if path.is_file():
            if self._is_image_file(path):
                return [str(path)]
            else:
                raise ValueError(f"File {path} is not a supported image format")
        
        # If it's a directory, get all images from it
        if path.is_dir():
            return self._get_images_from_directory(path)
        
        raise ValueError(f"Path {path} does not exist or is neither a file nor directory")
    
    def _is_image_file(self, path: Path) -> bool:
        """
        Check if a file is a supported image format
        
        Args:
            path: Path to check
            
        Returns:
            True if file is a supported image format
        """
        supported_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        return path.suffix.lower() in supported_extensions
    
    def _get_images_from_directory(self, directory: Path) -> list:
        """
        Get all image paths from a directory
        
        Args:
            directory: Directory path
            
        Returns:
            List of image file paths (as strings)
        """
        import glob
        
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
        image_paths = []
        for ext in extensions:
            image_paths.extend(glob.glob(str(directory / ext)))
            image_paths.extend(glob.glob(str(directory / ext.upper())))
        return sorted(image_paths)
