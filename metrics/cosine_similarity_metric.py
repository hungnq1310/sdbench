"""
Cosine Similarity Metric
Đánh giá tính nhất quán của cùng một ID ảo khi thay đổi góc chụp/ánh sáng
"""
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import glob

from .base_metric import BaseMetric


class CosineSimilarityMetric(BaseMetric):
    """
    Metric B: Đánh giá tính nhất quán
    Cosine Similarity > 0.7: Đạt chuẩn (cùng một người)
    Cosine Similarity 0.5-0.7: Cần kiểm tra lại
    Cosine Similarity < 0.5: Thất bại (model coi là 2 người khác nhau)
    
    Requires: FaceNetModel from models package
    """
    
    def __init__(self, model: Optional[Any] = None, config: Dict[str, Any] = None):
        """
        Initialize Cosine Similarity metric
        
        Args:
            model: FaceNetModel instance for face embedding extraction
            config: Configuration dictionary
        """
        super().__init__(model, config)
        
        if self.model is None:
            raise ValueError("CosineSimilarityMetric requires a FaceNetModel instance. "
                           "Pass model=FaceNetModel() when creating CosineSimilarityMetric.")
        

    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First embedding vector
            vec2: Second embedding vector
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    

    
    def compute(self, real_path: Path, fake_path: Path, **kwargs) -> Dict[str, Any]:
        """
        Compute Cosine Similarity
        
        Args:
            real_path: Path to real images (not used for this metric, but kept for consistency)
            fake_path: Path to fake/generated images directory
            **kwargs: 
                - image_pairs: List of tuples [(img1_path, img2_path), ...] to compare
                  If not provided, will compare all images pairwise
                - same_id_pairs: List of tuples indicating which pairs are same ID
                
        Returns:
            Dictionary with similarity scores and statistics
        """
        print(f"\n{'='*60}")
        print(f"Computing Cosine Similarity Metric (Metric B)")
        print(f"Model: {self.model.get_model_name()}")
        print(f"{'='*60}")
        
        image_pairs = kwargs.get('image_pairs', None)
        
        # If no specific pairs provided, use images from fake_path
        if image_pairs is None:
            real_images = self.get_image_paths(real_path)
            fake_images = self.get_image_paths(fake_path)
            print(f"\nFound {len(fake_images)} images in fake directory")
            
            if len(fake_images) < 2 or len(real_images) < 2:
                raise ValueError("Need at least 2 images to compute similarity")
            
            # Create pairs: compare first image with all others
            assert len(fake_images) == len(real_images), "Mismatch in number of fake and real images"
            # sort both lists to ensure consistent pairing
            fake_images.sort()
            real_images.sort()
            image_pairs = [(fake_images[i], real_images[i]) for i in range(len(fake_images))]
            print(f"Created {len(image_pairs)} pairs for comparison")
        
        similarities = []
        pair_details = []
        
        print("\n--- Computing similarities ---")
        for i, (img1_path, img2_path) in enumerate(image_pairs):
            try:
                print(f"Pair {i+1}/{len(image_pairs)}: {Path(img1_path).name} vs {Path(img2_path).name}")
                
                # Use model's extract_embeddings method
                emb1 = self.model.extract_embeddings(Path(img1_path)).flatten()
                emb2 = self.model.extract_embeddings(Path(img2_path)).flatten()
                
                similarity = self._cosine_similarity(emb1, emb2)
                similarities.append(similarity)
                
                pair_details.append({
                    'image1': str(Path(img1_path).name),
                    'image2': str(Path(img2_path).name),
                    'similarity': similarity,
                    'interpretation': self._interpret_single_score(similarity)
                })
                
                print(f"  Similarity: {similarity:.4f} - {self._interpret_single_score(similarity)}")
                
            except Exception as e:
                print(f"  Error processing pair: {e}")
                continue
        
        if not similarities:
            raise ValueError("No valid similarity scores computed")
        
        # Calculate statistics
        avg_similarity = np.mean(similarities)
        std_similarity = np.std(similarities)
        min_similarity = np.min(similarities)
        max_similarity = np.max(similarities)
        
        self.results = {
            'metric': 'Cosine Similarity',
            'average_similarity': avg_similarity,
            'std_similarity': std_similarity,
            'min_similarity': min_similarity,
            'max_similarity': max_similarity,
            'num_pairs': len(similarities),
            'pair_details': pair_details,
            'model': self.model.get_model_name(),
            'interpretation': self.interpret_result({
                'average_similarity': avg_similarity,
                'min_similarity': min_similarity
            })
        }
        
        print(f"\n{'='*60}")
        print(f"Results:")
        print(f"  Average Similarity: {avg_similarity:.4f}")
        print(f"  Std Dev: {std_similarity:.4f}")
        print(f"  Min: {min_similarity:.4f}")
        print(f"  Max: {max_similarity:.4f}")
        print(f"\nInterpretation: {self.results['interpretation']}")
        print(f"{'='*60}")
        
        return self.results
    
    def _interpret_single_score(self, score: float) -> str:
        """Interpret a single similarity score"""
        if score > 0.7:
            return "✓ Đạt chuẩn (Same person)"
        elif score >= 0.5:
            return "⚠ Cần kiểm tra (Borderline)"
        else:
            return "✗ Thất bại (Different person)"
    
    def interpret_result(self, result: Dict[str, Any]) -> str:
        """
        Interpret overall cosine similarity results
        
        Thresholds:
        - > 0.7: Đạt chuẩn (cùng một người, độ tin cậy cực cao)
        - 0.5-0.7: Cần kiểm tra lại (cùng người nhưng có biến thiên)
        - < 0.5: Thất bại (model coi là 2 người khác nhau)
        """
        avg_score = result.get('average_similarity', 0)
        min_score = result.get('min_similarity', 0)
        
        if avg_score > 0.7 and min_score > 0.5:
            return f"✓ ĐẠT CHUẨN (Avg: {avg_score:.3f}): Cùng một người, độ tin cậy cực cao. Ảnh sinh giữ được tính nhất quán ID."
        elif avg_score >= 0.5:
            return f"⚠ CẦN KIỂM TRA (Avg: {avg_score:.3f}, Min: {min_score:.3f}): Cùng người nhưng có biến thiên. Cần đánh giá kỹ hơn."
        else:
            return f"✗ THẤT BẠI (Avg: {avg_score:.3f}): Model coi các ảnh là người khác nhau. ID không nhất quán."
