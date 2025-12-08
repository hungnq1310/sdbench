"""
FaceNet Model for face recognition and similarity
"""
import torch
import numpy as np
from pathlib import Path
from typing import Any, List, Union
from PIL import Image
from torchvision import transforms

from .base_model import BaseModel


class FaceNetModel(BaseModel):
    """
    FaceNet model for face recognition and similarity measurement
    Uses InceptionResnetV1 or ResNet50 as fallback
    """
    
    def __init__(self, device: str = 'auto', use_facenet: bool = True):
        """
        Initialize FaceNet model
        
        Args:
            device: Device to run on
            use_facenet: Try to use facenet-pytorch library (fallback to ResNet50 if not available)
        """
        self.use_facenet = use_facenet
        self.model_type = None  # Will be set in _load_model
        super().__init__(device)
    
    def _load_model(self):
        """Load FaceNet or fallback model"""
        if self.use_facenet:
            try:
                from facenet_pytorch import InceptionResnetV1
                print("Loading FaceNet (InceptionResnetV1) model...")
                self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
                self.model_type = 'facenet'
                print(f"FaceNet loaded on {self.device}")
                return
            except ImportError:
                print("facenet-pytorch not found, falling back to ResNet50...")
        
        # Fallback to ResNet50
        from torchvision.models import resnet50, ResNet50_Weights
        print("Loading ResNet50 model...")
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.fc = torch.nn.Identity()  # Remove classification layer
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model_type = 'resnet50'
        print(f"ResNet50 loaded on {self.device}")
    
    def preprocess(self, image_input: Union[Path, Image.Image, List[Path]]) -> torch.Tensor:
        """
        Preprocess image(s) for FaceNet/ResNet
        
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
        
        # FaceNet uses 160x160, ResNet uses 224x224
        size = 160 if self.model_type == 'facenet' else 224
        
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        tensors = [transform(img) for img in images]
        batch = torch.stack(tensors)
        
        return batch
    
    def inference(self, preprocessed_input: torch.Tensor) -> torch.Tensor:
        """
        Run FaceNet/ResNet inference
        
        Args:
            preprocessed_input: Preprocessed tensor from preprocess()
            
        Returns:
            Embedding tensor
        """
        with torch.no_grad():
            preprocessed_input = preprocessed_input.to(self.device)
            embeddings = self.model(preprocessed_input)
        return embeddings
    
    def postprocess(self, model_output: torch.Tensor, output_type: str = 'embedding') -> np.ndarray:
        """
        Postprocess FaceNet/ResNet output
        
        Args:
            model_output: Raw embeddings from inference()
            output_type: 'embedding' (only supported type)
            
        Returns:
            Normalized numpy array of embeddings
        """
        if output_type != 'embedding':
            raise ValueError(f"FaceNet only supports 'embedding' output type, got '{output_type}'")
        
        # Convert to numpy
        embeddings = model_output.cpu().numpy()
        
        # Normalize embeddings (L2 normalization)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)  # Avoid division by zero
        
        return embeddings
