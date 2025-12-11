"""
t-SNE Visualization Metric
Đánh giá khả năng phân tách các ID nhân vật ảo khác nhau
"""
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
import glob
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from .base_metric import BaseMetric


class TSNEMetric(BaseMetric):
    """
    Metric C: Đánh giá khả năng phân tách
    Sử dụng t-SNE để visualize embedding space.
    Các ảnh của cùng một ID phải tụ lại thành cụm riêng biệt và cách xa các ID khác.
    
    Requires: FaceNetModel from models package
    """
    
    def __init__(self, model: Optional[Any] = None, config: Dict[str, Any] = None):
        """
        Initialize t-SNE metric
        
        Args:
            model: FaceNetModel instance for face embedding extraction
            config: Configuration dictionary
        """
        super().__init__(model, config)
        
        if self.model is None:
            raise ValueError("TSNEMetric requires a FaceNetModel instance. "
                           "Pass model=FaceNetModel() when creating TSNEMetric.")
        

    
    def _get_image_paths_by_id(self, directory: Path, id_pattern: Optional[str] = None) -> Dict[str, List[Path]]:
        """
        Organize images by ID
        
        Args:
            directory: Directory containing images
            id_pattern: Pattern to extract ID from filename (e.g., 'id_{id}_*.jpg')
                       If None, assumes each subdirectory is an ID
                       
        Returns:
            Dictionary mapping ID to list of image paths
        """
        images_by_id = {}
        
        # Check if directory has subdirectories (each subdirectory is an ID)
        subdirs = [d for d in Path(directory).iterdir() if d.is_dir()]
        
        if subdirs:
            # Use subdirectories as IDs
            for subdir in subdirs:
                id_name = subdir.name
                extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
                image_paths = []
                for ext in extensions:
                    image_paths.extend(list(subdir.glob(ext)))
                    image_paths.extend(list(subdir.glob(ext.upper())))
                
                if image_paths:
                    images_by_id[id_name] = sorted(image_paths)
        else:
            # All images in one directory - try to extract ID from filename
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp']
            all_images = []
            for ext in extensions:
                all_images.extend(list(Path(directory).glob(ext)))
                all_images.extend(list(Path(directory).glob(ext.upper())))
            
            # Group by ID (assume filename format: id_XXX_*.jpg or similar)
            for img_path in all_images:
                # Try to extract ID from filename
                parts = img_path.stem.split('_')
                if len(parts) >= 2 and parts[0] == 'id':
                    id_name = parts[1]
                else:
                    # Default: use first part of filename as ID
                    id_name = parts[0] if parts else 'default'
                
                if id_name not in images_by_id:
                    images_by_id[id_name] = []
                images_by_id[id_name].append(img_path)
        
        return images_by_id
    
    def _calculate_cluster_metrics(self, embeddings: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate clustering quality metrics
        
        Returns:
            Dictionary with intra-cluster and inter-cluster distances
        """
        unique_labels = np.unique(labels)
        
        # Intra-cluster distances (within same ID)
        intra_distances = []
        for label in unique_labels:
            mask = labels == label
            cluster_embeddings = embeddings[mask]
            if len(cluster_embeddings) > 1:
                # Calculate pairwise distances within cluster
                from scipy.spatial.distance import pdist
                distances = pdist(cluster_embeddings, metric='euclidean')
                intra_distances.extend(distances)
        
        # Inter-cluster distances (between different IDs)
        inter_distances = []
        for i, label1 in enumerate(unique_labels):
            for label2 in unique_labels[i+1:]:
                mask1 = labels == label1
                mask2 = labels == label2
                emb1 = embeddings[mask1]
                emb2 = embeddings[mask2]
                
                # Calculate distances between all pairs
                from scipy.spatial.distance import cdist
                distances = cdist(emb1, emb2, metric='euclidean')
                inter_distances.extend(distances.flatten())
        
        # turn into float instead of type np.array
        return {
            'avg_intra_cluster_distance': float(np.mean(intra_distances)) if intra_distances else 0,
            'avg_inter_cluster_distance': float(np.mean(inter_distances)) if inter_distances else 0,
            'separation_ratio': (float(np.mean(inter_distances)) / float(np.mean(intra_distances))) if intra_distances and inter_distances else 0
        }
    
    def compute(self, real_path: Path, fake_path: Path, **kwargs) -> Dict[str, Any]:
        """
        Compute t-SNE visualization and clustering metrics
        
        Args:
            real_path: Path to real images (optional, can be None)
            fake_path: Path to fake/generated images directory
            **kwargs:
                - output_plot: Path to save visualization plot
                - perplexity: t-SNE perplexity parameter (default: 30)
                - n_iter: Number of iterations for t-SNE (default: 1000)
                
        Returns:
            Dictionary with clustering metrics and visualization path
        """
        print(f"\n{'='*60}")
        print(f"Computing t-SNE Metric (Metric C)")
        print(f"Model: {self.model.get_model_name()}")
        print(f"{'='*60}")
        
        output_plot = kwargs.get('output_plot', 'output/tsne_visualization.png')
        perplexity = kwargs.get('perplexity', 30)
        n_iter = kwargs.get('n_iter', 1000)
        
        # Get images organized by ID
        print("\n--- Organizing images by ID ---")
        images_by_id = self._get_image_paths_by_id(Path(fake_path))
        
        print(f"Found {len(images_by_id)} IDs:")
        for id_name, img_list in images_by_id.items():
            print(f"  ID '{id_name}': {len(img_list)} images")
        
        if len(images_by_id) < 2:
            raise ValueError("Need at least 2 different IDs for t-SNE analysis")
        
        # Extract embeddings
        print("\n--- Extracting embeddings ---")
        all_embeddings = []
        all_labels = []
        all_image_names = []
        
        for id_name, img_paths in images_by_id.items():
            for img_path in img_paths:
                try:
                    # Use model's extract_embeddings method
                    embedding = self.model.extract_embeddings(img_path).flatten()
                    all_embeddings.append(embedding)
                    all_labels.append(id_name)
                    all_image_names.append(img_path.name)
                    print(f"  Processed: {img_path.name}")
                except Exception as e:
                    print(f"  Error processing {img_path.name}: {e}")
        
        if len(all_embeddings) < 2:
            raise ValueError("Not enough valid embeddings extracted")
        
        embeddings = np.array(all_embeddings)
        labels = np.array(all_labels)
        
        print(f"\nTotal embeddings: {len(embeddings)}")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        
        # Apply t-SNE
        print(f"\n--- Applying t-SNE (perplexity={perplexity}, n_iter={n_iter}) ---")
        tsne = TSNE(n_components=2, perplexity=min(perplexity, len(embeddings)-1), 
                    max_iter=n_iter, random_state=42, verbose=1)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        # Calculate clustering metrics
        print("\n--- Calculating clustering metrics ---")
        cluster_metrics = self._calculate_cluster_metrics(embeddings, labels)
        
        # Create visualization
        print("\n--- Creating visualization ---")
        self._create_visualization(embeddings_2d, labels, all_image_names, output_plot)
        
        # Interpret results
        interpretation = self._interpret_clustering(cluster_metrics, len(images_by_id))
        
        self.results = {
            'metric': 't-SNE',
            'num_ids': len(images_by_id),
            'total_images': len(embeddings),
            'images_per_id': {id_name: len(imgs) for id_name, imgs in images_by_id.items()},
            'cluster_metrics': cluster_metrics,
            'visualization_path': str(output_plot),
            'model': self.model.get_model_name(),
            'interpretation': interpretation
        }
        
        print(f"\n{'='*60}")
        print(f"Results:")
        print(f"  Number of IDs: {len(images_by_id)}")
        print(f"  Total images: {len(embeddings)}")
        print(f"  Avg intra-cluster distance: {cluster_metrics['avg_intra_cluster_distance']:.4f}")
        print(f"  Avg inter-cluster distance: {cluster_metrics['avg_inter_cluster_distance']:.4f}")
        print(f"  Separation ratio: {cluster_metrics['separation_ratio']:.4f}")
        print(f"  Visualization saved to: {output_plot}")
        print(f"\nInterpretation: {interpretation}")
        print(f"{'='*60}")
        
        return self.results
    
    def _create_visualization(self, embeddings_2d: np.ndarray, labels: np.ndarray, 
                             image_names: List[str], output_path: str):
        """Create t-SNE visualization plot"""
        plt.figure(figsize=(12, 10))
        
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       c=[color], label=f'ID: {label}', alpha=0.7, s=100)
        
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.title('t-SNE Visualization of Face Embeddings by ID', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to: {output_path}")
    
    def _interpret_clustering(self, metrics: Dict[str, float], num_ids: int) -> str:
        """
        Interpret clustering quality
        
        Separation ratio > 2.0: Excellent separation (các ID tách biệt rõ ràng)
        Separation ratio 1.5-2.0: Good separation
        Separation ratio < 1.5: Poor separation (các ID bị overlap)
        """
        sep_ratio = metrics['separation_ratio']
        
        if sep_ratio > 2.0:
            return (f"✓ ĐẠT CHUẨN - Excellent (Separation Ratio = {sep_ratio:.2f}): "
                   f"Các {num_ids} ID phân tách rõ ràng, không bị overlap. "
                   f"Mỗi ID có đặc điểm sinh trắc học riêng biệt.")
        elif sep_ratio >= 1.5:
            return (f"✓ Good (Separation Ratio = {sep_ratio:.2f}): "
                   f"Các ID phân tách tốt với ít overlap.")
        else:
            return (f"✗ THẤT BẠI - Poor (Separation Ratio = {sep_ratio:.2f}): "
                   f"Các ID bị overlap nhiều, khó phân biệt. Cần cải thiện chất lượng ID generation.")
    
    def interpret_result(self, result: Dict[str, Any]) -> str:
        """Interpret t-SNE results"""
        return result.get('interpretation', 'No interpretation available')
