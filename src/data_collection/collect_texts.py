"""
Main data collection script.
Orchestrates collection from both Anthroposophy and Vedanta sources.
"""

import argparse
from collect_anthroposophy import AnthroposophyCollector
from collect_vedanta import VedantaCollector
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Collect texts for Anthroposophy-Vedanta comparison project'
    )
    parser.add_argument(
        '--source',
        choices=['anthroposophy', 'vedanta', 'all'],
        default='all',
        help='Which texts to collect'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Starting text collection")
    logger.info("=" * 60)
    
    if args.source in ['anthroposophy', 'all']:
        logger.info("\n📚 Collecting Anthroposophy texts...")
        anthro_collector = AnthroposophyCollector()
        anthro_collector.run_collection()
    
    if args.source in ['vedanta', 'all']:
        logger.info("\n📚 Collecting Vedanta texts...")
        vedanta_collector = VedantaCollector()
        vedanta_collector.run_collection()
    
    logger.info("\n" + "=" * 60)
    logger.info("Collection complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
