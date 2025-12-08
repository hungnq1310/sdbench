"""
Abstract base class for all models
"""
from abc import ABC, abstractmethod
from typing import Any, List, Union
from pathlib import Path
import numpy as np
from PIL import Image


class BaseModel(ABC):
    """
    Abstract base class for models used in metrics evaluation
    Supports both embedding extraction and direct scoring
    """
    
    def __init__(self, device: str = 'auto'):
        """
        Initialize model
        
        Args:
            device: Device to run model on ('cuda', 'cpu', or 'auto')
        """
        import torch
        
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model = None
        self._load_model()
    
    @abstractmethod
    def _load_model(self):
        """
        Load the pretrained model
        Should set self.model
        """
        pass
    
    @abstractmethod
    def preprocess(self, image_input: Union[Path, Image.Image, List[Path]]) -> Any:
        """
        Preprocess image(s) for model input
        
        Args:
            image_input: Single image path, PIL Image, or list of image paths
            
        Returns:
            Preprocessed tensor(s) ready for model
        """
        pass
    
    @abstractmethod
    def inference(self, preprocessed_input: Any) -> Any:
        """
        Run model inference
        
        Args:
            preprocessed_input: Output from preprocess()
            
        Returns:
            Raw model output
        """
        pass
    
    @abstractmethod
    def postprocess(self, model_output: Any, output_type: str = 'embedding') -> Union[np.ndarray, float]:
        """
        Postprocess model output
        
        Args:
            model_output: Raw output from inference()
            output_type: Type of output desired
                - 'embedding': Extract feature embeddings (for similarity/clustering)
                - 'score': Direct scoring (for discriminator-like models) - NOT IMPLEMENTED YET
                
        Returns:
            Processed output as numpy array (embedding) or float (score)
        """
        pass
    
    def extract_embeddings(self, image_input: Union[Path, Image.Image, List[Path]]) -> np.ndarray:
        """
        Extract embeddings from image(s) - convenience method
        
        Args:
            image_input: Single image or list of images
            
        Returns:
            Embedding vector(s) as numpy array
        """
        preprocessed = self.preprocess(image_input)
        raw_output = self.inference(preprocessed)
        embeddings = self.postprocess(raw_output, output_type='embedding')
        return embeddings
    
    def compute_score(self, image_input: Union[Path, Image.Image, List[Path]]) -> float:
        """
        Compute direct score from image(s) - for discriminator-like models
        NOT IMPLEMENTED YET - placeholder for future use
        
        Args:
            image_input: Single image or list of images
            
        Returns:
            Score value
        """
        raise NotImplementedError("Direct scoring not implemented yet. Use extract_embeddings() instead.")
    
    def get_model_name(self) -> str:
        """
        Get the name of the model
        
        Returns:
            Model name
        """
        return self.__class__.__name__
