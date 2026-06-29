#!/usr/bin/env python3
"""
Script to run the CLV prediction pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path so package imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.clv_pipeline import CLVPipeline


def setup_logging(log_level: str = 'INFO'):
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('clv_pipeline.log')
        ]
    )


def main():
    """Main function to run the CLV pipeline."""
    parser = argparse.ArgumentParser(description='Run CLV Prediction Pipeline')
    parser.add_argument('--config', '-c', default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--log-level', '-l', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    parser.add_argument('--output-dir', '-o', default='models/',
                       help='Output directory for results')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("Starting CLV prediction pipeline")

    try:
        # Initialize and run pipeline
        pipeline = CLVPipeline(args.config)

        # Override output directory if specified
        if args.output_dir != 'models/':
            pipeline.config['output_dir'] = args.output_dir

        # Run the pipeline
        results = pipeline.run()

        # Print summary
        print("\n" + "="*50)
        print("CLV PREDICTION PIPELINE RESULTS")
        print("="*50)
        print(f"Total Customers: {results['total_customers']}")
        print(f"Average CLV: {results['average_clv']:.2f}")
        print(f"Total CLV: {results['total_clv']:.2f}")
        print(f"Average Churn Risk: {results['average_churn_risk']:.3f}")
        print("\nSegment Distribution:")
        for segment, count in results['segment_distribution'].items():
            print(f"  {segment}: {count}")
        print("\nTop 5 Priority Customers:")
        for i, customer in enumerate(results['top_priority_customers'][:5], 1):
            print(f"  {i}. {customer}")
        print("="*50)

        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
