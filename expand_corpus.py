"""
Enhanced data collection with working URLs and multiple sources.
"""

import requests
from pathlib import Path
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_text(url, output_path, description):
    """Download a text file from URL."""
    try:
        logger.info(f"Downloading: {description}")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        logger.info(f"✓ Saved: {output_path.name}")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to download {description}: {e}")
        return False


def collect_vedanta_texts():
    """Collect Vedanta texts from multiple sources."""
    base_dir = Path("data/vedanta")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    texts = [
        # Bhagavad Gita - Multiple translations
        {
            "url": "https://www.gutenberg.org/cache/epub/2388/pg2388.txt",
            "file": "bhagavad_gita_besant.txt",
            "desc": "Bhagavad Gita (Besant translation)"
        },
        {
            "url": "https://www.gutenberg.org/files/4351/4351-0.txt",
            "file": "bhagavad_gita_arnold.txt",
            "desc": "Bhagavad Gita - Song Celestial (Arnold)"
        },
        
        # Upanishads
        {
            "url": "https://www.gutenberg.org/cache/epub/3283/pg3283.txt",
            "file": "upanishads_muller_part1.txt",
            "desc": "Upanishads Part 1 (Max Müller)"
        },
        {
            "url": "https://www.gutenberg.org/cache/epub/3182/pg3182.txt",
            "file": "upanishads_muller_part2.txt",
            "desc": "Upanishads Part 2 (Max Müller)"
        },
        
        # Vedanta Philosophy
        {
            "url": "https://www.gutenberg.org/cache/epub/1435/pg1435.txt",
            "file": "raja_yoga_vivekananda.txt",
            "desc": "Raja Yoga by Swami Vivekananda"
        },
        {
            "url": "https://www.gutenberg.org/cache/epub/1437/pg1437.txt",
            "file": "karma_yoga_vivekananda.txt",
            "desc": "Karma Yoga by Swami Vivekananda"
        },
        {
            "url": "https://www.gutenberg.org/cache/epub/1436/pg1436.txt",
            "file": "jnana_yoga_vivekananda.txt",
            "desc": "Jnana Yoga by Swami Vivekananda"
        },
        {
            "url": "https://www.gutenberg.org/cache/epub/1454/pg1454.txt",
            "file": "bhakti_yoga_vivekananda.txt",
            "desc": "Bhakti Yoga by Swami Vivekananda"
        },
    ]
    
    successful = 0
    for text in texts:
        if download_text(text["url"], base_dir / text["file"], text["desc"]):
            successful += 1
            time.sleep(2)  # Be respectful to servers
    
    logger.info(f"\n✓ Successfully downloaded {successful}/{len(texts)} Vedanta texts")
    return successful


def collect_anthroposophy_texts():
    """Collect Anthroposophy texts."""
    base_dir = Path("data/anthroposophy")
    base_dir.mkdir(parents=True, exist_ok=True)
    
    texts = [
        # Rudolf Steiner works on Project Gutenberg
        {
            "url": "https://www.gutenberg.org/cache/epub/7143/pg7143.txt",
            "file": "steiner_christianity_mystical_fact.txt",
            "desc": "Christianity as Mystical Fact - Rudolf Steiner"
        },
        {
            "url": "https://www.gutenberg.org/cache/epub/4399/pg4399.txt",
            "file": "steiner_philosophy_of_freedom.txt",
            "desc": "The Philosophy of Freedom - Rudolf Steiner"
        },
        {
            "url": "https://www.gutenberg.org/files/54260/54260-0.txt",
            "file": "steiner_spiritual_guidance.txt",
            "desc": "Spiritual Guidance of Man and Mankind - Rudolf Steiner"
        },
        {
            "url": "https://www.gutenberg.org/cache/epub/53569/pg53569.txt",
            "file": "steiner_goethes_weltanschauung.txt",
            "desc": "Goethes Weltanschauung - Rudolf Steiner"
        },
        {
            "url": "https://www.gutenberg.org/files/59257/59257-0.txt",
            "file": "steiner_threefold_commonwealth.txt",
            "desc": "The Threefold Commonwealth - Rudolf Steiner"
        },
    ]
    
    successful = 0
    for text in texts:
        if download_text(text["url"], base_dir / text["file"], text["desc"]):
            successful += 1
            time.sleep(2)
    
    logger.info(f"\n✓ Successfully downloaded {successful}/{len(texts)} Anthroposophy texts")
    return successful


def main():
    logger.info("="*70)
    logger.info("EXPANDING TEXT CORPUS")
    logger.info("="*70)
    
    logger.info("\n📚 Collecting Vedanta Texts...")
    vedanta_count = collect_vedanta_texts()
    
    logger.info("\n📚 Collecting Anthroposophy Texts...")
    anthro_count = collect_anthroposophy_texts()
    
    logger.info("\n" + "="*70)
    logger.info(f"✅ COLLECTION COMPLETE")
    logger.info(f"   Vedanta texts: {vedanta_count}")
    logger.info(f"   Anthroposophy texts: {anthro_count}")
    logger.info(f"   Total: {vedanta_count + anthro_count}")
    logger.info("="*70)


if __name__ == "__main__":
    main()
