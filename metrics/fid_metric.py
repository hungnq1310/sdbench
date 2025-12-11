"""
FID (Fréchet Inception Distance) Metric
Đánh giá độ chân thực của ảnh sinh so với ảnh thật
"""
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from scipy import linalg
import glob

from .base_metric import BaseMetric


class FIDMetric(BaseMetric):
    """
    Metric A: Đánh giá độ chân thực
    FID càng thấp (<50) thì ảnh sinh càng giống ảnh thật
    
    Requires: InceptionModel from models package
    """
    
    def __init__(self, model: Optional[Any] = None, config: Dict[str, Any] = None):
        """
        Initialize FID metric
        
        Args:
            model: InceptionModel instance for feature extraction
            config: Configuration dictionary
        """
        super().__init__(model, config)
        
        if self.model is None:
            raise ValueError("FIDMetric requires an InceptionModel instance. "
                           "Pass model=InceptionModel() when creating FIDMetric.")
        

    

    
    def _extract_features_batch(self, image_paths: list, batch_size: int = 32) -> np.ndarray:
        """
        Extract features from images using the provided model
        
        Args:
            image_paths: List of image paths
            batch_size: Batch size for processing
            
        Returns:
            Feature array of shape (num_images, feature_dim)
        """
        features = []
        num_batches = (len(image_paths) + batch_size - 1) // batch_size
        
        print(f"Extracting features from {len(image_paths)} images...")
        
        for i in range(num_batches):
            batch_paths = image_paths[i * batch_size:(i + 1) * batch_size]
            
            # Filter out invalid images
            valid_paths = []
            for img_path in batch_paths:
                try:
                    from PIL import Image
                    Image.open(img_path).convert('RGB')
                    valid_paths.append(img_path)
                except Exception as e:
                    print(f"Warning: Failed to open {img_path}: {e}")
                    continue
            
            if valid_paths:
                # Use model's extract_embeddings method
                batch_features = self.model.extract_embeddings(valid_paths)
                features.append(batch_features)
            
            print(f"Progress: {min((i + 1) * batch_size, len(image_paths))}/{len(image_paths)}")
        
        return np.vstack(features)
    
    def _calculate_fid(self, features_real: np.ndarray, features_fake: np.ndarray) -> float:
        """
        Calculate FID score between real and fake features
        
        FID = ||mu_real - mu_fake||^2 + Tr(Sigma_real + Sigma_fake - 2*sqrt(Sigma_real*Sigma_fake))
        """
        # Calculate mean and covariance
        mu_real = np.mean(features_real, axis=0)
        mu_fake = np.mean(features_fake, axis=0)
        
        sigma_real = np.cov(features_real, rowvar=False)
        sigma_fake = np.cov(features_fake, rowvar=False)
        
        # Calculate FID
        diff = mu_real - mu_fake
        
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)
        
        if not np.isfinite(covmean).all():
            print("Warning: FID calculation produced singular product; adding epsilon")
            offset = np.eye(sigma_real.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma_real + offset).dot(sigma_fake + offset))
        
        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {m}")
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2 * covmean)
        
        return float(fid)
    
    def compute(self, real_path: Path, fake_path: Path, **kwargs) -> Dict[str, Any]:
        """
        Compute FID score
        
        Args:
            real_path: Path to real images directory
            fake_path: Path to fake/generated images directory
            
        Returns:
            Dictionary with FID score and statistics
        """
        print(f"\n{'='*60}")
        print(f"Computing FID Metric (Metric A)")
        print(f"Model: {self.model.get_model_name()}")
        print(f"{'='*60}")
        
        # Get image paths (supports directories, single files, or lists of files)
        real_images = self.get_image_paths(real_path)
        fake_images = self.get_image_paths(fake_path)
        
        print(f"\nFound {len(real_images)} real images")
        print(f"Found {len(fake_images)} fake images")
        
        if len(real_images) == 0 or len(fake_images) == 0:
            raise ValueError("No images found in one or both directories")
        
        # FID requires at least 2 images to calculate covariance matrix
        if len(real_images) < 2 or len(fake_images) < 2:
            raise ValueError(
                f"FID metric requires at least 2 images in each set to calculate covariance matrix.\n"
                f"Found: {len(real_images)} real images, {len(fake_images)} fake images.\n"
                f"Please provide at least 2 images for both real and fake sets."
            )
        
        # sort image paths to ensure consistent ordering
        real_images.sort()
        fake_images.sort()
        
        # Extract features using injected model
        print("\n--- Extracting features from REAL images ---")
        features_real = self._extract_features_batch(real_images)
        
        print("\n--- Extracting features from FAKE images ---")
        features_fake = self._extract_features_batch(fake_images)
        
        # Calculate FID
        print("\n--- Calculating FID score ---")
        fid_score = self._calculate_fid(features_real, features_fake)
        
        self.results = {
            'metric': 'FID',
            'score': fid_score,
            'num_real_images': len(real_images),
            'num_fake_images': len(fake_images),
            'model': self.model.get_model_name(),
            'interpretation': self.interpret_result({'score': fid_score})
        }
        
        print(f"\nFID Score: {fid_score:.2f}")
        print(f"Interpretation: {self.results['interpretation']}")
        
        return self.results
    
    def interpret_result(self, result: Dict[str, Any]) -> str:
        """
        Interpret FID score
        
        FID thresholds:
        - < 50: Excellent (ảnh sinh rất giống ảnh thật)
        - 50-100: Good (chất lượng tốt)
        - 100-200: Acceptable (chấp nhận được)
        - > 200: Poor (kém, cần cải thiện)
        """
        score = result['score']
        
        if score < 50:
            return f"✓ ĐẠT CHUẨN - Excellent (FID = {score:.2f}): Ảnh sinh rất giống ảnh thật về mặt thống kê"
        elif score < 100:
            return f"✓ Good (FID = {score:.2f}): Chất lượng tốt, ảnh sinh có độ chân thực cao"
        elif score < 200:
            return f"⚠ Acceptable (FID = {score:.2f}): Chấp nhận được nhưng cần xem xét cải thiện"
        else:
            return f"✗ THẤT BẠI - Poor (FID = {score:.2f}): Ảnh sinh khác biệt nhiều so với ảnh thật"
