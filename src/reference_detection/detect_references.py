"""
Reference detection module for identifying mentions of Vedanta concepts in Anthroposophy texts.
"""

import re
from pathlib import Path
from typing import List, Dict, Set, Tuple
import json
from collections import defaultdict, Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReferenceDetector:
    """Detect references to Vedanta concepts, texts, and figures in Anthroposophy texts."""
    
    def __init__(self):
        """Initialize with reference dictionaries."""
        
        # Vedanta texts
        self.vedanta_texts = {
            'bhagavad gita', 'bhagavad-gita', 'bhagavadgita', 'gita',
            'upanishads', 'upanishad', 'isha upanishad', 'kena upanishad',
            'katha upanishad', 'prashna upanishad', 'mundaka upanishad',
            'mandukya upanishad', 'taittiriya upanishad', 'aitareya upanishad',
            'chandogya upanishad', 'brihadaranyaka upanishad', 
            'svetasvatara upanishad', 'kaushitaki upanishad',
            'brahma sutras', 'brahma-sutras', 'vedanta sutras',
            'vedas', 'rig veda', 'sama veda', 'yajur veda', 'atharva veda'
        }
        
        # Vedanta figures
        self.vedanta_figures = {
            'shankara', 'shankaracharya', 'adi shankara', 'adi shankaracharya',
            'vyasa', 'veda vyasa', 'krishna', 'lord krishna', 'sri krishna',
            'arjuna', 'yajnavalkya', 'janaka', 'ramana maharshi',
            'vivekananda', 'swami vivekananda', 'ramakrishna'
        }
        
        # Key Vedanta concepts
        self.vedanta_concepts = {
            'vedanta', 'advaita', 'advaita vedanta', 'non-dualism', 'non-duality',
            'brahman', 'atman', 'atma', 'maya', 'moksha', 'moksa',
            'karma', 'dharma', 'samsara', 'nirvana', 'samadhi',
            'self-realization', 'self realization', 'self-knowledge',
            'jnana', 'bhakti', 'yoga', 'raja yoga', 'karma yoga',
            'bhakti yoga', 'jnana yoga', 'kundalini', 'chakra', 'chakras',
            'prana', 'om', 'aum', 'tat tvam asi', 'aham brahmasmi',
            'sat-chit-ananda', 'sat chit ananda', 'turiya'
        }
        
        # Sanskrit terms often used
        self.sanskrit_terms = {
            'purusha', 'prakriti', 'gunas', 'sattva', 'rajas', 'tamas',
            'jiva', 'jivatman', 'paramatman', 'ishvara', 'saguna brahman',
            'nirguna brahman', 'manas', 'buddhi', 'chitta', 'avidya',
            'vidya', 'viveka', 'vairagya', 'sadhana', 'tapas', 'satsang'
        }
        
        # Combine all terms
        self.all_terms = (
            self.vedanta_texts | 
            self.vedanta_figures | 
            self.vedanta_concepts | 
            self.sanskrit_terms
        )
        
    def find_term_occurrences(
        self, 
        text: str, 
        terms: Set[str],
        context_chars: int = 100
    ) -> List[Dict]:
        """
        Find all occurrences of specified terms in text.
        
        Args:
            text: Text to search
            terms: Set of terms to find
            context_chars: Characters of context to capture around match
            
        Returns:
            List of dictionaries with match information
        """
        occurrences = []
        text_lower = text.lower()
        
        for term in terms:
            # Create regex pattern with word boundaries
            pattern = r'\b' + re.escape(term) + r'\b'
            
            for match in re.finditer(pattern, text_lower):
                start = match.start()
                end = match.end()
                
                # Extract context
                context_start = max(0, start - context_chars)
                context_end = min(len(text), end + context_chars)
                context = text[context_start:context_end]
                
                occurrences.append({
                    'term': term,
                    'position': start,
                    'matched_text': match.group(),
                    'context': context.strip()
                })
        
        return occurrences
    
    def analyze_text(self, text: str, title: str = "Unknown") -> Dict:
        """
        Analyze a text for Vedanta references.
        
        Args:
            text: Text to analyze
            title: Title of the text
            
        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Analyzing: {title}")
        
        # Find all references
        all_occurrences = self.find_term_occurrences(text, self.all_terms)
        
        # Categorize occurrences
        texts_found = [occ for occ in all_occurrences if occ['term'] in self.vedanta_texts]
        figures_found = [occ for occ in all_occurrences if occ['term'] in self.vedanta_figures]
        concepts_found = [occ for occ in all_occurrences if occ['term'] in self.vedanta_concepts]
        sanskrit_found = [occ for occ in all_occurrences if occ['term'] in self.sanskrit_terms]
        
        # Count frequencies
        term_frequencies = Counter([occ['term'] for occ in all_occurrences])
        
        analysis = {
            'title': title,
            'total_references': len(all_occurrences),
            'unique_terms': len(term_frequencies),
            'categories': {
                'texts': len(texts_found),
                'figures': len(figures_found),
                'concepts': len(concepts_found),
                'sanskrit_terms': len(sanskrit_found)
            },
            'term_frequencies': dict(term_frequencies.most_common()),
            'occurrences_by_category': {
                'texts': texts_found,
                'figures': figures_found,
                'concepts': concepts_found,
                'sanskrit_terms': sanskrit_found
            },
            'top_terms': term_frequencies.most_common(10)
        }
        
        return analysis
    
    def analyze_file(self, filepath: Path) -> Dict:
        """
        Analyze a file for Vedanta references.
        
        Args:
            filepath: Path to text file
            
        Returns:
            Analysis dictionary
        """
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
        
        return self.analyze_text(text, title=filepath.name)
    
    def analyze_corpus(self, corpus_dir: Path, output_path: Path = None) -> Dict:
        """
        Analyze entire corpus for Vedanta references.
        
        Args:
            corpus_dir: Directory with text files
            output_path: Optional path to save results
            
        Returns:
            Combined analysis results
        """
        corpus_dir = Path(corpus_dir)
        file_analyses = []
        
        # Analyze each file
        for filepath in corpus_dir.glob("*.txt"):
            try:
                analysis = self.analyze_file(filepath)
                file_analyses.append(analysis)
            except Exception as e:
                logger.error(f"Error analyzing {filepath}: {e}")
        
        # Aggregate statistics
        total_references = sum(a['total_references'] for a in file_analyses)
        all_terms = Counter()
        
        for analysis in file_analyses:
            all_terms.update(analysis['term_frequencies'])
        
        corpus_analysis = {
            'corpus_name': corpus_dir.name,
            'num_files': len(file_analyses),
            'total_references': total_references,
            'unique_terms_across_corpus': len(all_terms),
            'most_common_terms': all_terms.most_common(20),
            'file_analyses': file_analyses
        }
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(corpus_analysis, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {output_path}")
        
        return corpus_analysis
    
    def generate_report(self, analysis: Dict) -> str:
        """
        Generate human-readable report from analysis.
        
        Args:
            analysis: Analysis dictionary
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append("VEDANTA REFERENCE DETECTION REPORT")
        report.append("=" * 70)
        report.append(f"\nCorpus: {analysis['corpus_name']}")
        report.append(f"Files analyzed: {analysis['num_files']}")
        report.append(f"Total references found: {analysis['total_references']}")
        report.append(f"Unique terms found: {analysis['unique_terms_across_corpus']}")
        
        report.append("\n\nMOST FREQUENTLY MENTIONED TERMS:")
        report.append("-" * 70)
        for term, count in analysis['most_common_terms']:
            report.append(f"  {term:30s} : {count:4d} occurrences")
        
        report.append("\n\nPER-FILE ANALYSIS:")
        report.append("-" * 70)
        
        for file_analysis in analysis['file_analyses']:
            if file_analysis['total_references'] > 0:
                report.append(f"\n{file_analysis['title']}")
                report.append(f"  Total references: {file_analysis['total_references']}")
                report.append(f"  Unique terms: {file_analysis['unique_terms']}")
                
                if file_analysis['top_terms']:
                    report.append("  Top terms:")
                    for term, count in file_analysis['top_terms'][:5]:
                        report.append(f"    - {term}: {count}")
        
        return "\n".join(report)


def main():
    """Main execution for reference detection."""
    detector = ReferenceDetector()
    
    # Analyze Anthroposophy corpus for Vedanta references
    logger.info("Analyzing Anthroposophy texts for Vedanta references...")
    
    anthro_dir = Path("data/anthroposophy")
    if anthro_dir.exists():
        analysis = detector.analyze_corpus(
            corpus_dir=anthro_dir,
            output_path=Path("results/vedanta_references_in_anthroposophy.json")
        )
        
        # Generate and save report
        report = detector.generate_report(analysis)
        report_path = Path("results/vedanta_references_report.txt")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(report)
        logger.info(f"Report saved to: {report_path}")
    else:
        logger.error(f"Directory not found: {anthro_dir}")


if __name__ == "__main__":
    main()
