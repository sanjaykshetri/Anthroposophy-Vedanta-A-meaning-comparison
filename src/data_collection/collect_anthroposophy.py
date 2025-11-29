"""
Data collection module for Anthroposophy texts.
Collects Rudolf Steiner's works and other anthroposophical literature.
"""

import requests
from bs4 import BeautifulSoup
import time
import os
from pathlib import Path
import json
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnthroposophyCollector:
    """Collect Anthroposophy texts from various online sources."""
    
    def __init__(self, output_dir: str = "data/anthroposophy"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
    def collect_rsarchive_texts(self) -> List[Dict]:
        """
        Collect texts from Rudolf Steiner Archive.
        Note: This is a template - actual URLs need to be verified.
        """
        texts = []
        
        # Example list of Rudolf Steiner's major works
        works = [
            "Philosophy of Freedom",
            "Theosophy",
            "Occult Science",
            "Knowledge of Higher Worlds",
            "Christianity as Mystical Fact"
        ]
        
        logger.info("Starting collection from Rudolf Steiner Archive")
        logger.info("Note: This requires adapting to actual source structure")
        
        return texts
    
    def collect_from_gutenberg(self) -> List[Dict]:
        """
        Collect Steiner texts available on Project Gutenberg.
        """
        texts = []
        
        # Rudolf Steiner works on Project Gutenberg
        gutenberg_ids = {
            "Christianity as Mystical Fact": "7143",
            "The Philosophy of Freedom": "4399",
        }
        
        for title, book_id in gutenberg_ids.items():
            try:
                url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
                logger.info(f"Fetching: {title}")
                
                response = requests.get(url, headers=self.headers, timeout=30)
                response.raise_for_status()
                
                text_data = {
                    'title': title,
                    'author': 'Rudolf Steiner',
                    'source': 'Project Gutenberg',
                    'book_id': book_id,
                    'text': response.text
                }
                
                texts.append(text_data)
                
                # Save individual file
                filename = f"steiner_{book_id}_{title.replace(' ', '_')}.txt"
                filepath = self.output_dir / filename
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                logger.info(f"Saved: {filename}")
                time.sleep(2)  # Be respectful to the server
                
            except Exception as e:
                logger.error(f"Error fetching {title}: {e}")
        
        return texts
    
    def collect_sacred_texts(self) -> List[Dict]:
        """
        Collect texts from sacred-texts.com.
        """
        texts = []
        
        # Example URLs - need to be verified and expanded
        urls = [
            "https://sacred-texts.com/eso/sta/index.htm",  # Example
        ]
        
        logger.info("Collecting from sacred-texts.com")
        logger.info("Note: Requires specific URL mapping")
        
        return texts
    
    def save_metadata(self, texts: List[Dict]) -> None:
        """Save metadata about collected texts."""
        metadata = {
            'collection_date': time.strftime('%Y-%m-%d'),
            'total_texts': len(texts),
            'texts': [
                {
                    'title': t['title'],
                    'author': t.get('author', 'Unknown'),
                    'source': t.get('source', 'Unknown'),
                    'text_length': len(t.get('text', ''))
                }
                for t in texts
            ]
        }
        
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
    
    def run_collection(self) -> None:
        """Run the complete collection process."""
        all_texts = []
        
        # Collect from different sources
        all_texts.extend(self.collect_from_gutenberg())
        all_texts.extend(self.collect_rsarchive_texts())
        all_texts.extend(self.collect_sacred_texts())
        
        # Save metadata
        self.save_metadata(all_texts)
        
        logger.info(f"Collection complete. Total texts: {len(all_texts)}")


def main():
    """Main execution function."""
    collector = AnthroposophyCollector()
    collector.run_collection()


if __name__ == "__main__":
    main()
