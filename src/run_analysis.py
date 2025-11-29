"""
Main analysis pipeline orchestrator.
"""

import argparse
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_collection.collect_texts import main as collect_texts_main
from preprocessing.text_cleaner import TextPreprocessor
from reference_detection.detect_references import ReferenceDetector
from analysis.similarity_analysis import SimilarityAnalyzer
from analysis.concept_mapping import ConceptMapper

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline(args):
    """
    Run the complete analysis pipeline.
    
    Args:
        args: Command-line arguments
    """
    logger.info("=" * 80)
    logger.info("ANTHROPOSOPHY-VEDANTA COMPARISON ANALYSIS PIPELINE")
    logger.info("=" * 80)
    
    # Step 1: Data Collection (if requested)
    if args.collect:
        logger.info("\n📚 Step 1: Collecting texts...")
        # Note: Implement proper collection
        logger.info("Data collection module ready (run collect_texts.py)")
    
    # Step 2: Text Preprocessing
    if args.preprocess:
        logger.info("\n🔧 Step 2: Preprocessing texts...")
        preprocessor = TextPreprocessor()
        
        # Process Anthroposophy texts
        anthro_dir = Path("data/anthroposophy")
        if anthro_dir.exists():
            preprocessor.preprocess_corpus(
                input_dir=anthro_dir,
                output_dir=Path("data/processed/anthroposophy")
            )
        
        # Process Vedanta texts
        vedanta_dir = Path("data/vedanta")
        if vedanta_dir.exists():
            preprocessor.preprocess_corpus(
                input_dir=vedanta_dir,
                output_dir=Path("data/processed/vedanta")
            )
    
    # Step 3: Reference Detection
    if args.references:
        logger.info("\n🔍 Step 3: Detecting Vedanta references...")
        detector = ReferenceDetector()
        
        anthro_dir = Path("data/anthroposophy")
        if anthro_dir.exists() and any(anthro_dir.glob("*.txt")):
            analysis = detector.analyze_corpus(
                corpus_dir=anthro_dir,
                output_path=Path("results/vedanta_references.json")
            )
            
            report = detector.generate_report(analysis)
            with open("results/vedanta_references_report.txt", 'w') as f:
                f.write(report)
            
            logger.info("Reference detection complete!")
        else:
            logger.warning(f"No texts found in {anthro_dir}")
    
    # Step 4: Similarity Analysis
    if args.similarity:
        logger.info("\n📊 Step 4: Computing similarity metrics...")
        analyzer = SimilarityAnalyzer()
        analyzer.run_full_analysis()
    
    # Step 5: Concept Mapping
    if args.concepts:
        logger.info("\n🗺️  Step 5: Mapping parallel concepts...")
        mapper = ConceptMapper()
        mapper.analyze_concepts()
    
    logger.info("\n" + "=" * 80)
    logger.info("✅ PIPELINE COMPLETE!")
    logger.info("=" * 80)
    logger.info("\nResults saved in: results/")
    logger.info("Check the following files:")
    logger.info("  - vedanta_references_report.txt")
    logger.info("  - similarity_analysis.json")
    logger.info("  - distinctive_terms.json")
    logger.info("  - topic_modeling.json")
    logger.info("  - concept_mappings.json")
    logger.info("  - similarity_heatmap.png")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Run Anthroposophy-Vedanta comparison analysis'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all analysis steps'
    )
    parser.add_argument(
        '--collect',
        action='store_true',
        help='Collect texts from online sources'
    )
    parser.add_argument(
        '--preprocess',
        action='store_true',
        help='Preprocess and clean texts'
    )
    parser.add_argument(
        '--references',
        action='store_true',
        help='Detect Vedanta references in Anthroposophy texts'
    )
    parser.add_argument(
        '--similarity',
        action='store_true',
        help='Compute similarity metrics'
    )
    parser.add_argument(
        '--concepts',
        action='store_true',
        help='Map parallel concepts'
    )
    
    args = parser.parse_args()
    
    # If --all is specified, run everything
    if args.all:
        args.collect = True
        args.preprocess = True
        args.references = True
        args.similarity = True
        args.concepts = True
    
    # If no specific steps specified, show help
    if not any([args.collect, args.preprocess, args.references, 
                args.similarity, args.concepts]):
        parser.print_help()
        return
    
    run_pipeline(args)


if __name__ == "__main__":
    main()
