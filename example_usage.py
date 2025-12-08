"""
Example script showing how to use individual metrics programmatically
"""

from pathlib import Path
from metrics import FIDMetric, CosineSimilarityMetric, TSNEMetric
from models import InceptionModel, FaceNetModel


def example_fid():
    """Example: Run FID metric"""
    print("\n=== Example: FID Metric ===")
    
    # Initialize model
    inception_model = InceptionModel(device='auto', image_size=1024)
    
    # Initialize metric with model
    metric = FIDMetric(model=inception_model)
    result = metric.compute(
        real_path=Path('./datahub/real_images'),
        fake_path=Path('./datahub/fake_images')
    )
    
    print(f"FID Score: {result['score']:.2f}")
    print(f"Model: {result['model']}")
    print(f"Interpretation: {result['interpretation']}")
    
    # Save results
    metric.save_results(Path('./output/example_fid.json'))


def example_cosine_similarity():
    """Example: Run Cosine Similarity metric"""
    print("\n=== Example: Cosine Similarity Metric ===")
    
    # Initialize model
    facenet_model = FaceNetModel(device='auto', use_facenet=True)
    
    # Initialize metric with model
    metric = CosineSimilarityMetric(model=facenet_model)
    
    # Option 1: Compare all images pairwise
    result = metric.compute(
        real_path=Path('./datahub/real_images'),
        fake_path=Path('./datahub/fake_images')
    )
    
    # Option 2: Specify specific image pairs
    # image_pairs = [
    #     ('./datahub/fake_images/id_001_frontal.jpg', './datahub/fake_images/id_001_side.jpg'),
    #     ('./datahub/fake_images/id_001_frontal.jpg', './datahub/fake_images/id_001_light.jpg'),
    # ]
    # result = metric.compute(
    #     real_path=Path('./datahub/real_images'),
    #     fake_path=Path('./datahub/fake_images'),
    #     image_pairs=image_pairs
    # )
    
    print(f"Average Similarity: {result['average_similarity']:.4f}")
    print(f"Model: {result['model']}")
    print(f"Interpretation: {result['interpretation']}")
    
    metric.save_results(Path('./output/example_cosine.json'))


def example_tsne():
    """Example: Run t-SNE metric"""
    print("\n=== Example: t-SNE Metric ===")
    
    # Initialize model
    facenet_model = FaceNetModel(device='auto', use_facenet=True)
    
    # Initialize metric with model
    metric = TSNEMetric(model=facenet_model)
    result = metric.compute(
        real_path=Path('./datahub/real_images'),
        fake_path=Path('./datahub/fake_images'),
        output_plot='./output/example_tsne.png',
        perplexity=30,
        n_iter=1000
    )
    
    print(f"Number of IDs: {result['num_ids']}")
    print(f"Separation Ratio: {result['cluster_metrics']['separation_ratio']:.2f}")
    print(f"Model: {result['model']}")
    print(f"Interpretation: {result['interpretation']}")
    print(f"Visualization saved to: {result['visualization_path']}")
    
    metric.save_results(Path('./output/example_tsne.json'))


if __name__ == '__main__':
    # Run individual examples
    
    # Example 1: FID
    try:
        example_fid()
    except Exception as e:
        print(f"Error in FID example: {e}")
    
    # Example 2: Cosine Similarity
    try:
        example_cosine_similarity()
    except Exception as e:
        print(f"Error in Cosine Similarity example: {e}")
    
    # Example 3: t-SNE
    try:
        example_tsne()
    except Exception as e:
        print(f"Error in t-SNE example: {e}")
