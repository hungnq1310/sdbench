"""
Main evaluation script for AI-generated vs Real images metrics

Usage:
    python evaluate.py --real-path ./datahub/real_images --fake-path ./datahub/fake_images --metrics all
    python evaluate.py --real-path ./datahub/real_images --fake-path ./datahub/fake_images --metrics fid
    python evaluate.py --real-path ./datahub/real_images --fake-path ./datahub/fake_images --metrics cosine tsne
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from metrics import FIDMetric, CosineSimilarityMetric, TSNEMetric, BaseMetric
from models import InceptionModel, FaceNetModel

INCEPTION_MODEL = InceptionModel(device='auto', image_size=1024)
FACENET_MODEL = FaceNetModel(device='auto', use_facenet=True)

class MetricEvaluator:
    """
    Main evaluator class to run multiple metrics
    """
    
    def __init__(self, real_path, fake_path, output_dir: Path = None):
        """
        Initialize evaluator
        
        Args:
            real_path: Path(s) to real images - can be:
                - Single directory path (str or Path)
                - Single image file path (str or Path)
                - List of image file paths
            fake_path: Path(s) to fake images - can be:
                - Single directory path (str or Path)
                - Single image file path (str or Path)
                - List of image file paths
            output_dir: Directory to save results (default: ./output)
        """
        # Store paths as-is (can be str, Path, or list)
        self.real_path = real_path
        self.fake_path = fake_path
        self.output_dir = Path(output_dir) if output_dir else Path('./output')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate paths exist (basic validation)
        self._validate_paths(self.real_path, "Real images")
        self._validate_paths(self.fake_path, "Fake images")
        
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def _validate_paths(self, path_or_paths, label: str):
        """Validate that provided paths exist"""
        if isinstance(path_or_paths, (list, tuple)):
            for p in path_or_paths:
                p = Path(p)
                if not p.exists():
                    raise ValueError(f"{label} path does not exist: {p}")
        else:
            p = Path(path_or_paths)
            if not p.exists():
                raise ValueError(f"{label} path does not exist: {p}")
        
    def run_fid(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run FID metric (Metric A: Đánh giá độ chân thực)
        
        Args:
            config: Configuration for FID metric
            
        Returns:
            FID results dictionary
        """
        print("\n" + "="*80)
        print("METRIC A: FID - Đánh giá độ chân thực")
        print("="*80)
        
        metric = FIDMetric(model=INCEPTION_MODEL, config=config)
        result = metric.compute(self.real_path, self.fake_path)
        
        # Save individual metric results
        output_file = self.output_dir / f"fid_results_{self.timestamp}.json"
        metric.save_results(output_file)
        return result
    
    def run_cosine_similarity(self, config: Dict[str, Any] = None, 
                             image_pairs: List[tuple] = None) -> Dict[str, Any]:
        """
        Run Cosine Similarity metric (Metric B: Đánh giá tính nhất quán)
        
        Args:
            config: Configuration for Cosine Similarity metric
            image_pairs: List of image pairs to compare
            
        Returns:
            Cosine Similarity results dictionary
        """
        print("\n" + "="*80)
        print("METRIC B: Cosine Similarity - Đánh giá tính nhất quán")
        print("="*80)
        
        metric = CosineSimilarityMetric(model=INCEPTION_MODEL, config=config)
        result = metric.compute(self.real_path, self.fake_path, image_pairs=image_pairs)
        
        # Save individual metric results
        output_file = self.output_dir / f"cosine_similarity_results_{self.timestamp}.json"
        metric.save_results(output_file)
        
        return result
    
    def run_tsne(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run t-SNE metric (Metric C: Đánh giá khả năng phân tách)
        
        Args:
            config: Configuration for t-SNE metric
            
        Returns:
            t-SNE results dictionary
        """
        print("\n" + "="*80)
        print("METRIC C: t-SNE - Đánh giá khả năng phân tách")
        print("="*80)
        
        metric = TSNEMetric(model=FACENET_MODEL, config=config)
        output_plot = self.output_dir / f"tsne_visualization_{self.timestamp}.png"
        result = metric.compute(self.real_path, self.fake_path, output_plot=str(output_plot))
        
        # Save individual metric results
        output_file = self.output_dir / f"tsne_results_{self.timestamp}.json"
        metric.save_results(output_file)
        
        return result
    
    def run_all_metrics(self, configs: Dict[str, Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run all metrics
        
        Args:
            configs: Dictionary of configurations for each metric
            
        Returns:
            Combined results from all metrics
        """
        configs = configs or {}
        
        print("\n" + "#"*80)
        print("# RUNNING ALL METRICS")
        print(f"# Real images: {self.real_path}")
        print(f"# Fake images: {self.fake_path}")
        print(f"# Output directory: {self.output_dir}")
        print("#"*80)
        
        all_results = {
            'timestamp': self.timestamp,
            'real_path': str(self.real_path),
            'fake_path': str(self.fake_path),
            'metrics': {}
        }
        
        # Run FID
        try:
            all_results['metrics']['fid'] = self.run_fid(configs.get('fid'))
        except Exception as e:
            print(f"\n✗ Error running FID metric: {e}")
            all_results['metrics']['fid'] = {'error': str(e)}
        
        # Run Cosine Similarity
        try:
            all_results['metrics']['cosine_similarity'] = self.run_cosine_similarity(
                configs.get('cosine_similarity')
            )
        except Exception as e:
            print(f"\n✗ Error running Cosine Similarity metric: {e}")
            all_results['metrics']['cosine_similarity'] = {'error': str(e)}
        
        # Run t-SNE
        try:
            all_results['metrics']['tsne'] = self.run_tsne(configs.get('tsne'))
        except Exception as e:
            print(f"\n✗ Error running t-SNE metric: {e}")
            all_results['metrics']['tsne'] = {'error': str(e)}
        
        # Save combined results
        output_file = self.output_dir / f"all_metrics_results_{self.timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        
        print("\n" + "#"*80)
        print("# EVALUATION COMPLETE")
        print(f"# Results saved to: {output_file}")
        print("#"*80)
        
        # Print summary
        self._print_summary(all_results)
        
        return all_results
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print summary of all metrics"""
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        
        metrics = results.get('metrics', {})
        
        # FID Summary
        if 'fid' in metrics and 'error' not in metrics['fid']:
            fid = metrics['fid']
            print(f"\n[Metric A] FID Score: {fid.get('score', 'N/A'):.2f}")
            print(f"  {fid.get('interpretation', 'N/A')}")
        
        # Cosine Similarity Summary
        if 'cosine_similarity' in metrics and 'error' not in metrics['cosine_similarity']:
            cos = metrics['cosine_similarity']
            print(f"\n[Metric B] Cosine Similarity: {cos.get('average_similarity', 'N/A'):.4f}")
            print(f"  {cos.get('interpretation', 'N/A')}")
        
        # t-SNE Summary
        if 'tsne' in metrics and 'error' not in metrics['tsne']:
            tsne = metrics['tsne']
            sep_ratio = tsne.get('cluster_metrics', {}).get('separation_ratio', 'N/A')
            print(f"\n[Metric C] t-SNE Separation Ratio: {str(sep_ratio) if isinstance(sep_ratio, float) else sep_ratio}")
            print(f"  {tsne.get('interpretation', 'N/A')}")
        
        print("\n" + "="*80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Evaluate AI-generated vs Real images using multiple metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all metrics with directories
  python evaluate.py --real-path ./datahub/real_images --fake-path ./datahub/fake_images
  
  # Run with specific image files
  python evaluate.py --real-path img1.jpg img2.jpg --fake-path gen1.jpg gen2.jpg --metrics fid
  
  # Mix directory and files
  python evaluate.py --real-path ./datahub/real_images --fake-path gen1.jpg gen2.jpg
  
  # Run only FID metric
  python evaluate.py --real-path ./datahub/real_images --fake-path ./datahub/fake_images --metrics fid
  
  # Specify output directory
  python evaluate.py --real-path ./datahub/real_images --fake-path ./datahub/fake_images --output ./my_results
        """
    )
    
    parser.add_argument('--real-path', type=str, nargs='+', required=True,
                       help='Path(s) to real images: directory, single file, or multiple files')
    parser.add_argument('--fake-path', type=str, nargs='+', required=True,
                       help='Path(s) to fake images: directory, single file, or multiple files')
    parser.add_argument('--output', type=str, default='./output',
                       help='Output directory for results (default: ./output)')
    parser.add_argument('--metrics', nargs='+', 
                       choices=['fid', 'cosine', 'tsne', 'all'],
                       default=['all'],
                       help='Metrics to run (default: all)')
    
    args = parser.parse_args()
    
    # Handle single or multiple paths
    real_path = args.real_path[0] if len(args.real_path) == 1 else args.real_path
    fake_path = args.fake_path[0] if len(args.fake_path) == 1 else args.fake_path
    
    # Initialize evaluator
    evaluator = MetricEvaluator(
        real_path=real_path,
        fake_path=fake_path,
        output_dir=args.output
    )
    
    # Determine which metrics to run
    metrics_to_run = args.metrics
    if 'all' in metrics_to_run:
        evaluator.run_all_metrics()
    else:
        results = {
            'timestamp': evaluator.timestamp,
            'real_path': str(evaluator.real_path),
            'fake_path': str(evaluator.fake_path),
            'metrics': {}
        }
        
        if 'fid' in metrics_to_run:
            results['metrics']['fid'] = evaluator.run_fid()
        
        if 'cosine' in metrics_to_run:
            results['metrics']['cosine_similarity'] = evaluator.run_cosine_similarity()
        
        if 'tsne' in metrics_to_run:
            results['metrics']['tsne'] = evaluator.run_tsne()
        
        # Save results
        output_file = evaluator.output_dir / f"selected_metrics_results_{evaluator.timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        evaluator._print_summary(results)


if __name__ == '__main__':
    main()
