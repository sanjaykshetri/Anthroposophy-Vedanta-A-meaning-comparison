"""
Data collection module for Vedanta texts.
Collects Upanishads, Bhagavad Gita, and Shankara's teachings.
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


class VedantaCollector:
    """Collect Vedanta texts from various online sources."""
    
    def __init__(self, output_dir: str = "data/vedanta"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
    def collect_bhagavad_gita(self) -> List[Dict]:
        """
        Collect Bhagavad Gita texts from various sources.
        """
        texts = []
        
        # Project Gutenberg - Bhagavad Gita
        gutenberg_ids = {
            "Bhagavad Gita (Besant translation)": "2388",
            "Bhagavad Gita (Arnold translation - Song Celestial)": "4351",
        }
        
        for title, book_id in gutenberg_ids.items():
            try:
                url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
                logger.info(f"Fetching: {title}")
                
                response = requests.get(url, headers=self.headers, timeout=30)
                response.raise_for_status()
                
                text_data = {
                    'title': title,
                    'category': 'Bhagavad Gita',
                    'source': 'Project Gutenberg',
                    'book_id': book_id,
                    'text': response.text
                }
                
                texts.append(text_data)
                
                # Save individual file
                filename = f"gita_{book_id}_{title.replace(' ', '_')[:50]}.txt"
                filepath = self.output_dir / filename
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(response.text)
                
                logger.info(f"Saved: {filename}")
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error fetching {title}: {e}")
        
        return texts
    
    def collect_upanishads(self) -> List[Dict]:
        """
        Collect Upanishads texts.
        """
        texts = []
        
        # Sacred-texts.com has comprehensive Upanishads collection
        # Major Upanishads to collect
        upanishads = [
            "Isha", "Kena", "Katha", "Prashna", "Mundaka", "Mandukya",
            "Taittiriya", "Aitareya", "Chandogya", "Brihadaranyaka",
            "Svetasvatara", "Kaushitaki", "Maitri"
        ]
        
        logger.info("Collecting Upanishads")
        logger.info(f"Target Upanishads: {', '.join(upanishads)}")
        
        # Sacred-texts URLs for Upanishads
        # Note: These URLs need to be verified and may need adjustment
        sacred_texts_base = "https://sacred-texts.com/hin/upan/"
        
        return texts
    
    def collect_brahma_sutras(self) -> List[Dict]:
        """
        Collect Brahma Sutras with Shankara's commentary.
        """
        texts = []
        
        logger.info("Collecting Brahma Sutras and Shankara's commentaries")
        
        # Sources for Brahma Sutras
        # Internet Archive and other digital libraries
        
        return texts
    
    def collect_shankara_works(self) -> List[Dict]:
        """
        Collect Adi Shankara's major works.
        """
        texts = []
        
        # Major works by Shankara
        works = [
            "Vivekachudamani (Crest-Jewel of Discrimination)",
            "Atmabodha (Self-Knowledge)",
            "Upadesa Sahasri (A Thousand Teachings)",
            "Aparokshanubhuti (Self-Realization)"
        ]
        
        logger.info(f"Collecting Shankara's works: {', '.join(works)}")
        
        return texts
    
    def collect_from_sacred_texts(self, path: str, title: str, category: str) -> Dict:
        """
        Helper function to collect individual texts from sacred-texts.com.
        
        Args:
            path: URL path on sacred-texts.com
            title: Title of the text
            category: Category (Upanishad, Gita, etc.)
        """
        try:
            url = f"https://sacred-texts.com{path}"
            logger.info(f"Fetching: {title} from {url}")
            
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract main content (adjust selector based on site structure)
            content = soup.find('body')
            if content:
                text = content.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)
            
            text_data = {
                'title': title,
                'category': category,
                'source': 'Sacred-texts.com',
                'url': url,
                'text': text
            }
            
            # Save individual file
            safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '_')).rstrip()
            filename = f"{category.lower().replace(' ', '_')}_{safe_title.replace(' ', '_')}.txt"
            filepath = self.output_dir / filename
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text)
            
            logger.info(f"Saved: {filename}")
            time.sleep(2)  # Be respectful to the server
            
            return text_data
            
        except Exception as e:
            logger.error(f"Error fetching {title}: {e}")
            return None
    
    def save_metadata(self, texts: List[Dict]) -> None:
        """Save metadata about collected texts."""
        metadata = {
            'collection_date': time.strftime('%Y-%m-%d'),
            'total_texts': len(texts),
            'categories': {},
            'texts': []
        }
        
        # Organize by category
        for text in texts:
            category = text.get('category', 'Unknown')
            if category not in metadata['categories']:
                metadata['categories'][category] = 0
            metadata['categories'][category] += 1
            
            metadata['texts'].append({
                'title': text['title'],
                'category': category,
                'source': text.get('source', 'Unknown'),
                'text_length': len(text.get('text', ''))
            })
        
        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Metadata saved to {metadata_path}")
    
    def run_collection(self) -> None:
        """Run the complete collection process."""
        all_texts = []
        
        # Collect from different sources
        all_texts.extend(self.collect_bhagavad_gita())
        all_texts.extend(self.collect_upanishads())
        all_texts.extend(self.collect_brahma_sutras())
        all_texts.extend(self.collect_shankara_works())
        
        # Save metadata
        self.save_metadata(all_texts)
        
        logger.info(f"Collection complete. Total texts: {len(all_texts)}")


def main():
    """Main execution function."""
    collector = VedantaCollector()
    collector.run_collection()


if __name__ == "__main__":
    main()
