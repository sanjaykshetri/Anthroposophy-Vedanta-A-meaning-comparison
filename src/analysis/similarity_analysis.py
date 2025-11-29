"""
Similarity analysis module for comparing Anthroposophy and Vedanta texts.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import json
import logging
from collections import defaultdict

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimilarityAnalyzer:
    """Analyze semantic similarity between Anthroposophy and Vedanta texts."""
    
    def __init__(self, processed_dir: Path = Path("data/processed")):
        """
        Initialize analyzer.
        
        Args:
            processed_dir: Directory containing processed texts
        """
        self.processed_dir = Path(processed_dir)
        self.anthro_texts = []
        self.vedanta_texts = []
        self.vectorizer = None
        self.tfidf_matrix = None
        
    def load_texts(self) -> Tuple[List[str], List[str]]:
        """
        Load processed texts from both corpora.
        
        Returns:
            Tuple of (anthroposophy_texts, vedanta_texts)
        """
        logger.info("Loading texts...")
        
        # Load Anthroposophy texts
        anthro_dir = self.processed_dir / "anthroposophy"
        if anthro_dir.exists():
            for filepath in anthro_dir.glob("*.txt"):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        self.anthro_texts.append({
                            'title': filepath.name,
                            'text': f.read()
                        })
                except Exception as e:
                    logger.error(f"Error loading {filepath}: {e}")
        
        # Load Vedanta texts
        vedanta_dir = self.processed_dir / "vedanta"
        if vedanta_dir.exists():
            for filepath in vedanta_dir.glob("*.txt"):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        self.vedanta_texts.append({
                            'title': filepath.name,
                            'text': f.read()
                        })
                except Exception as e:
                    logger.error(f"Error loading {filepath}: {e}")
        
        logger.info(f"Loaded {len(self.anthro_texts)} Anthroposophy texts")
        logger.info(f"Loaded {len(self.vedanta_texts)} Vedanta texts")
        
        return self.anthro_texts, self.vedanta_texts
    
    def compute_tfidf_similarity(self, max_features: int = 5000) -> Dict:
        """
        Compute TF-IDF similarity between all texts.
        
        Args:
            max_features: Maximum number of features for TF-IDF
            
        Returns:
            Dictionary with similarity results
        """
        logger.info("Computing TF-IDF similarity...")
        
        # Combine all texts
        all_texts = []
        labels = []
        
        for text_dict in self.anthro_texts:
            all_texts.append(text_dict['text'])
            labels.append(('anthroposophy', text_dict['title']))
        
        for text_dict in self.vedanta_texts:
            all_texts.append(text_dict['text'])
            labels.append(('vedanta', text_dict['title']))
        
        # Create TF-IDF matrix
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8,
            stop_words='english'
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        
        # Compute pairwise similarities
        similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
        # Find cross-corpus similarities (Anthroposophy vs Vedanta)
        n_anthro = len(self.anthro_texts)
        cross_similarities = similarity_matrix[:n_anthro, n_anthro:]
        
        # Find most similar pairs
        similar_pairs = []
        for i in range(n_anthro):
            for j in range(len(self.vedanta_texts)):
                similar_pairs.append({
                    'anthroposophy_text': self.anthro_texts[i]['title'],
                    'vedanta_text': self.vedanta_texts[j]['title'],
                    'similarity': float(cross_similarities[i, j])
                })
        
        # Sort by similarity
        similar_pairs.sort(key=lambda x: x['similarity'], reverse=True)
        
        results = {
            'num_anthroposophy_texts': n_anthro,
            'num_vedanta_texts': len(self.vedanta_texts),
            'average_similarity': float(np.mean(cross_similarities)),
            'max_similarity': float(np.max(cross_similarities)),
            'min_similarity': float(np.min(cross_similarities)),
            'most_similar_pairs': similar_pairs[:20],
            'similarity_matrix': cross_similarities.tolist()
        }
        
        logger.info(f"Average cross-corpus similarity: {results['average_similarity']:.4f}")
        
        return results
    
    def extract_distinctive_terms(self, top_n: int = 50) -> Dict:
        """
        Extract most distinctive terms for each corpus.
        
        Args:
            top_n: Number of top terms to extract
            
        Returns:
            Dictionary with distinctive terms
        """
        logger.info("Extracting distinctive terms...")
        
        if self.tfidf_matrix is None:
            self.compute_tfidf_similarity()
        
        n_anthro = len(self.anthro_texts)
        
        # Average TF-IDF for each corpus
        anthro_avg = np.array(self.tfidf_matrix[:n_anthro].mean(axis=0)).flatten()
        vedanta_avg = np.array(self.tfidf_matrix[n_anthro:].mean(axis=0)).flatten()
        
        # Get feature names
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        
        # Get top terms for each corpus
        anthro_top_indices = anthro_avg.argsort()[-top_n:][::-1]
        vedanta_top_indices = vedanta_avg.argsort()[-top_n:][::-1]
        
        # Compute distinctiveness (difference in TF-IDF)
        distinctiveness = np.abs(anthro_avg - vedanta_avg)
        distinctive_indices = distinctiveness.argsort()[-top_n:][::-1]
        
        results = {
            'anthroposophy_top_terms': [
                {'term': feature_names[i], 'tfidf': float(anthro_avg[i])}
                for i in anthro_top_indices
            ],
            'vedanta_top_terms': [
                {'term': feature_names[i], 'tfidf': float(vedanta_avg[i])}
                for i in vedanta_top_indices
            ],
            'most_distinctive_terms': [
                {
                    'term': feature_names[i],
                    'anthro_tfidf': float(anthro_avg[i]),
                    'vedanta_tfidf': float(vedanta_avg[i]),
                    'difference': float(distinctiveness[i])
                }
                for i in distinctive_indices
            ]
        }
        
        return results
    
    def perform_topic_modeling(self, n_topics: int = 10) -> Dict:
        """
        Perform topic modeling on combined corpus.
        
        Args:
            n_topics: Number of topics to extract
            
        Returns:
            Dictionary with topic modeling results
        """
        logger.info(f"Performing topic modeling with {n_topics} topics...")
        
        # Combine all texts
        all_texts = [t['text'] for t in self.anthro_texts + self.vedanta_texts]
        
        # Create count vectorizer for LDA
        count_vectorizer = CountVectorizer(
            max_features=1000,
            max_df=0.8,
            min_df=2,
            stop_words='english'
        )
        
        count_matrix = count_vectorizer.fit_transform(all_texts)
        
        # Fit LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            max_iter=20
        )
        
        lda_matrix = lda.fit_transform(count_matrix)
        
        # Get feature names
        feature_names = count_vectorizer.get_feature_names_out()
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            top_indices = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            topics.append({
                'topic_id': topic_idx,
                'top_words': top_words,
                'weights': [float(topic[i]) for i in top_indices]
            })
        
        # Analyze topic distribution by corpus
        n_anthro = len(self.anthro_texts)
        anthro_topics = lda_matrix[:n_anthro].mean(axis=0)
        vedanta_topics = lda_matrix[n_anthro:].mean(axis=0)
        
        results = {
            'n_topics': n_topics,
            'topics': topics,
            'anthroposophy_topic_distribution': anthro_topics.tolist(),
            'vedanta_topic_distribution': vedanta_topics.tolist()
        }
        
        return results
    
    def find_similar_passages(
        self,
        anthro_text: str,
        vedanta_text: str,
        window_size: int = 500,
        top_n: int = 10
    ) -> List[Dict]:
        """
        Find similar passages between two texts.
        
        Args:
            anthro_text: Anthroposophy text
            vedanta_text: Vedanta text
            window_size: Size of text window (characters)
            top_n: Number of top similar passages to return
            
        Returns:
            List of similar passage pairs
        """
        logger.info("Finding similar passages...")
        
        # Split into windows
        def create_windows(text, window_size):
            windows = []
            for i in range(0, len(text) - window_size, window_size // 2):
                windows.append(text[i:i + window_size])
            return windows
        
        anthro_windows = create_windows(anthro_text, window_size)
        vedanta_windows = create_windows(vedanta_text, window_size)
        
        # Vectorize windows
        all_windows = anthro_windows + vedanta_windows
        vectorizer = TfidfVectorizer(max_features=500)
        window_matrix = vectorizer.fit_transform(all_windows)
        
        # Compute similarities
        n_anthro_windows = len(anthro_windows)
        anthro_matrix = window_matrix[:n_anthro_windows]
        vedanta_matrix = window_matrix[n_anthro_windows:]
        
        similarities = cosine_similarity(anthro_matrix, vedanta_matrix)
        
        # Find top similar pairs
        similar_passages = []
        for i in range(len(anthro_windows)):
            for j in range(len(vedanta_windows)):
                similar_passages.append({
                    'similarity': float(similarities[i, j]),
                    'anthro_passage': anthro_windows[i],
                    'vedanta_passage': vedanta_windows[j],
                    'anthro_position': i * (window_size // 2),
                    'vedanta_position': j * (window_size // 2)
                })
        
        # Sort and return top N
        similar_passages.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similar_passages[:top_n]
    
    def visualize_similarity_matrix(
        self,
        similarity_matrix: np.ndarray,
        output_path: Path
    ) -> None:
        """
        Create heatmap visualization of similarity matrix.
        
        Args:
            similarity_matrix: Matrix of similarities
            output_path: Path to save visualization
        """
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            similarity_matrix,
            cmap='YlOrRd',
            cbar_kws={'label': 'Cosine Similarity'}
        )
        plt.title('Anthroposophy vs Vedanta Text Similarity')
        plt.xlabel('Vedanta Texts')
        plt.ylabel('Anthroposophy Texts')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Similarity matrix saved to: {output_path}")
    
    def run_full_analysis(self, output_dir: Path = Path("results")) -> None:
        """
        Run complete similarity analysis pipeline.
        
        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load texts
        self.load_texts()
        
        if not self.anthro_texts or not self.vedanta_texts:
            logger.error("No texts found. Please run data collection first.")
            return
        
        # TF-IDF similarity
        similarity_results = self.compute_tfidf_similarity()
        with open(output_dir / "similarity_analysis.json", 'w') as f:
            json.dump(similarity_results, f, indent=2)
        
        # Visualize similarity matrix
        self.visualize_similarity_matrix(
            np.array(similarity_results['similarity_matrix']),
            output_dir / "similarity_heatmap.png"
        )
        
        # Distinctive terms
        distinctive_terms = self.extract_distinctive_terms()
        with open(output_dir / "distinctive_terms.json", 'w') as f:
            json.dump(distinctive_terms, f, indent=2)
        
        # Topic modeling
        topic_results = self.perform_topic_modeling()
        with open(output_dir / "topic_modeling.json", 'w') as f:
            json.dump(topic_results, f, indent=2)
        
        logger.info(f"Analysis complete. Results saved to: {output_dir}")


def main():
    """Main execution for similarity analysis."""
    analyzer = SimilarityAnalyzer()
    analyzer.run_full_analysis()


if __name__ == "__main__":
    main()
