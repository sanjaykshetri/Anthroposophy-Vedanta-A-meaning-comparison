"""
Text preprocessing module for cleaning and normalizing collected texts.
"""

import re
import string
from pathlib import Path
from typing import List, Dict, Tuple
import json
import logging

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Preprocess and clean spiritual texts for analysis."""
    
    def __init__(self, language: str = 'english'):
        """
        Initialize preprocessor.
        
        Args:
            language: Language for stopwords and processing
        """
        self.language = language
        self.lemmatizer = WordNetLemmatizer()
        
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            logger.warning("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            self.nlp = None
        
        try:
            self.stop_words = set(stopwords.words(language))
        except LookupError:
            logger.warning("NLTK stopwords not found. Run: python -m nltk.downloader stopwords")
            self.stop_words = set()
        
        # Custom stopwords for spiritual texts (optional - may want to keep these)
        self.spiritual_stopwords = {
            'thee', 'thou', 'thy', 'thine', 'verily', 'thus', 'unto'
        }
        
    def remove_gutenberg_headers(self, text: str) -> str:
        """
        Remove Project Gutenberg headers and footers.
        
        Args:
            text: Raw text from Project Gutenberg
            
        Returns:
            Cleaned text without headers/footers
        """
        # Remove header
        start_markers = [
            r'\*\*\* START OF (THIS|THE) PROJECT GUTENBERG',
            r'START OF THE PROJECT GUTENBERG',
            r'^\*\*\*START\*\*\*'
        ]
        
        for marker in start_markers:
            match = re.search(marker, text, re.IGNORECASE | re.MULTILINE)
            if match:
                text = text[match.end():]
                break
        
        # Remove footer
        end_markers = [
            r'\*\*\* END OF (THIS|THE) PROJECT GUTENBERG',
            r'END OF THE PROJECT GUTENBERG',
            r'\*\*\*END\*\*\*'
        ]
        
        for marker in end_markers:
            match = re.search(marker, text, re.IGNORECASE)
            if match:
                text = text[:match.start()]
                break
        
        return text.strip()
    
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove Project Gutenberg artifacts
        text = self.remove_gutenberg_headers(text)
        
        # Remove special characters but keep Sanskrit diacritics if present
        # Keep alphanumeric, spaces, and common punctuation
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\n{3,}', '\n\n', text)  # Multiple newlines to double
        
        # Remove excessive whitespace
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\t+', ' ', text)
        
        # Remove page numbers and chapter markers (common patterns)
        text = re.sub(r'\n\d+\n', '\n', text)
        text = re.sub(r'\[pg \d+\]', '', text, flags=re.IGNORECASE)
        
        return text.strip()
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        try:
            sentences = sent_tokenize(text)
            return sentences
        except Exception as e:
            logger.error(f"Error in sentence tokenization: {e}")
            # Fallback to simple splitting
            return text.split('. ')
    
    def tokenize_words(self, text: str) -> List[str]:
        """
        Split text into word tokens.
        
        Args:
            text: Input text
            
        Returns:
            List of word tokens
        """
        try:
            tokens = word_tokenize(text)
            return tokens
        except Exception as e:
            logger.error(f"Error in word tokenization: {e}")
            # Fallback to simple splitting
            return text.split()
    
    def remove_stopwords(self, tokens: List[str], keep_spiritual: bool = True) -> List[str]:
        """
        Remove stopwords from token list.
        
        Args:
            tokens: List of word tokens
            keep_spiritual: Whether to keep spiritual-specific words
            
        Returns:
            Filtered token list
        """
        if keep_spiritual:
            stop_set = self.stop_words
        else:
            stop_set = self.stop_words | self.spiritual_stopwords
        
        filtered = [token for token in tokens if token.lower() not in stop_set]
        return filtered
    
    def lemmatize(self, tokens: List[str]) -> List[str]:
        """
        Lemmatize tokens.
        
        Args:
            tokens: List of word tokens
            
        Returns:
            Lemmatized token list
        """
        lemmatized = [self.lemmatizer.lemmatize(token.lower()) for token in tokens]
        return lemmatized
    
    def extract_named_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract named entities using spaCy.
        
        Args:
            text: Input text
            
        Returns:
            List of (entity_text, entity_label) tuples
        """
        if not self.nlp:
            logger.warning("spaCy not available for NER")
            return []
        
        doc = self.nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
    
    def preprocess_file(
        self,
        input_path: Path,
        output_path: Path,
        include_entities: bool = False
    ) -> Dict:
        """
        Preprocess a single text file.
        
        Args:
            input_path: Path to input file
            output_path: Path to save processed file
            include_entities: Whether to extract named entities
            
        Returns:
            Dictionary with processing statistics
        """
        logger.info(f"Processing: {input_path.name}")
        
        # Read file
        with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_text = f.read()
        
        # Clean text
        cleaned_text = self.clean_text(raw_text)
        
        # Tokenize
        sentences = self.tokenize_sentences(cleaned_text)
        words = self.tokenize_words(cleaned_text)
        
        # Remove punctuation and convert to lowercase
        words_cleaned = [
            word.lower() for word in words 
            if word.isalnum() or word in string.punctuation
        ]
        
        # Remove stopwords
        words_no_stop = self.remove_stopwords(words_cleaned)
        
        # Lemmatize
        words_lemmatized = self.lemmatize(words_no_stop)
        
        # Prepare output data
        processed_data = {
            'filename': input_path.name,
            'original_length': len(raw_text),
            'cleaned_length': len(cleaned_text),
            'num_sentences': len(sentences),
            'num_words': len(words),
            'num_unique_words': len(set(words_lemmatized)),
            'cleaned_text': cleaned_text,
            'sentences': sentences,
            'tokens': words_lemmatized
        }
        
        # Extract entities if requested
        if include_entities and self.nlp:
            entities = self.extract_named_entities(cleaned_text)
            processed_data['entities'] = entities
        
        # Save processed data
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        json_path = output_path.with_suffix('.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=2, ensure_ascii=False)
        
        # Save cleaned text
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_text)
        
        logger.info(f"Saved to: {output_path.name}")
        
        return processed_data
    
    def preprocess_corpus(
        self,
        input_dir: Path,
        output_dir: Path,
        file_pattern: str = "*.txt"
    ) -> List[Dict]:
        """
        Preprocess all files in a directory.
        
        Args:
            input_dir: Directory with raw texts
            output_dir: Directory to save processed texts
            file_pattern: Glob pattern for files to process
            
        Returns:
            List of processing statistics for each file
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_stats = []
        
        for input_file in input_dir.glob(file_pattern):
            output_file = output_dir / input_file.name
            
            try:
                stats = self.preprocess_file(input_file, output_file)
                all_stats.append(stats)
            except Exception as e:
                logger.error(f"Error processing {input_file}: {e}")
        
        # Save corpus-level statistics
        corpus_stats_path = output_dir / 'corpus_statistics.json'
        with open(corpus_stats_path, 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Corpus statistics saved to: {corpus_stats_path}")
        
        return all_stats


def main():
    """Main execution for preprocessing."""
    preprocessor = TextPreprocessor()
    
    # Process Anthroposophy texts
    logger.info("Processing Anthroposophy corpus...")
    anthro_stats = preprocessor.preprocess_corpus(
        input_dir=Path("data/anthroposophy"),
        output_dir=Path("data/processed/anthroposophy")
    )
    
    # Process Vedanta texts
    logger.info("Processing Vedanta corpus...")
    vedanta_stats = preprocessor.preprocess_corpus(
        input_dir=Path("data/vedanta"),
        output_dir=Path("data/processed/vedanta")
    )
    
    logger.info(f"Preprocessing complete!")
    logger.info(f"Anthroposophy texts: {len(anthro_stats)}")
    logger.info(f"Vedanta texts: {len(vedanta_stats)}")


if __name__ == "__main__":
    main()
