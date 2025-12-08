"""
InceptionV3 Model for FID metric
"""
import torch
import numpy as np
from pathlib import Path
from typing import Any, List, Union
from PIL import Image
from torchvision import transforms
from torchvision.models import inception_v3, Inception_V3_Weights

from .base_model import BaseModel


class InceptionModel(BaseModel):
    """
    InceptionV3 model for extracting image features
    Used primarily for FID (Fréchet Inception Distance) calculation
    """
    
    def __init__(self, device: str = 'auto', image_size: int = 1024):
        """
        Initialize InceptionV3 model
        
        Args:
            device: Device to run on
            image_size: Input image size before resizing to 299x299 (default: 1024)
        """
        self.image_size = image_size
        super().__init__(device)
    
    def _load_model(self):
        """Load pretrained InceptionV3 model"""
        print("Loading InceptionV3 model...")
        self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        # Remove final classification layer to get features
        self.model.fc = torch.nn.Identity()
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"InceptionV3 loaded on {self.device}")
    
    def preprocess(self, image_input: Union[Path, Image.Image, List[Path]]) -> torch.Tensor:
        """
        Preprocess image(s) for InceptionV3
        
        Args:
            image_input: Image path(s) or PIL Image
            
        Returns:
            Preprocessed tensor
        """
        # Handle single image
        if isinstance(image_input, (Path, str)):
            images = [Image.open(image_input).convert('RGB')]
        elif isinstance(image_input, Image.Image):
            images = [image_input.convert('RGB')]
        elif isinstance(image_input, list):
            images = [Image.open(img).convert('RGB') if isinstance(img, (Path, str)) else img.convert('RGB') 
                     for img in image_input]
        else:
            raise ValueError(f"Unsupported input type: {type(image_input)}")
        
        # Resize to target size first (e.g., 1024x1024), then to 299x299 for InceptionV3
        transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),  # First resize to standard size
            transforms.Resize((299, 299)),  # Then to InceptionV3 input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        tensors = [transform(img) for img in images]
        batch = torch.stack(tensors)
        
        return batch
    
    def inference(self, preprocessed_input: torch.Tensor) -> torch.Tensor:
        """
        Run InceptionV3 inference
        
        Args:
            preprocessed_input: Preprocessed tensor from preprocess()
            
        Returns:
            Feature tensor
        """
        with torch.no_grad():
            preprocessed_input = preprocessed_input.to(self.device)
            features = self.model(preprocessed_input)
        return features
    
    def postprocess(self, model_output: torch.Tensor, output_type: str = 'embedding') -> np.ndarray:
        """
        Postprocess InceptionV3 output
        
        Args:
            model_output: Raw features from inference()
            output_type: 'embedding' (only supported type for InceptionV3)
            
        Returns:
            Numpy array of features
        """
        if output_type != 'embedding':
            raise ValueError(f"InceptionV3 only supports 'embedding' output type, got '{output_type}'")
        
        # Convert to numpy and return
        embeddings = model_output.cpu().numpy()
        return embeddings
