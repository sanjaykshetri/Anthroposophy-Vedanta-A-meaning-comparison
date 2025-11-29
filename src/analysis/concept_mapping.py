"""
Concept mapping module for identifying parallel concepts between traditions.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import json
import logging
from collections import defaultdict

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConceptMapper:
    """Map parallel concepts between Anthroposophy and Vedanta."""
    
    def __init__(self):
        """Initialize concept mapper with predefined concept pairs."""
        
        # Hypothesized parallel concepts
        self.concept_pairs = {
            'anthroposophy': {
                'spiritual world': 'Higher spiritual realms beyond physical',
                'etheric body': 'Life force body',
                'astral body': 'Soul body, emotional nature',
                'ego': 'Individual self, I-consciousness',
                'supersensible': 'Beyond physical senses',
                'spiritual science': 'Scientific investigation of spiritual realms',
                'clairvoyance': 'Spiritual perception',
                'reincarnation': 'Successive earthly lives',
                'karma': 'Law of destiny and action',
                'initiation': 'Path to higher knowledge',
                'higher self': 'Divine aspect of individual',
                'lucifer': 'Force of pride and isolation',
                'ahriman': 'Force of materialism',
                'christ': 'Divine being of love and unity'
            },
            'vedanta': {
                'brahman': 'Ultimate reality, absolute consciousness',
                'atman': 'Individual soul, true Self',
                'maya': 'Illusion, appearance',
                'moksha': 'Liberation from cycle of rebirth',
                'samsara': 'Cycle of birth and death',
                'karma': 'Law of action and consequence',
                'dharma': 'Cosmic order, righteous duty',
                'jnana': 'Knowledge, wisdom path',
                'bhakti': 'Devotion, love path',
                'yoga': 'Union, spiritual practice',
                'samadhi': 'State of absorption',
                'turiya': 'Fourth state of consciousness',
                'sat-chit-ananda': 'Being-consciousness-bliss',
                'tat tvam asi': 'That thou art'
            }
        }
        
        # Load sentence transformer model if available
        self.model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded sentence transformer model")
            except Exception as e:
                logger.warning(f"Could not load model: {e}")
    
    def compute_concept_embeddings(self) -> Tuple[Dict, Dict]:
        """
        Compute embeddings for all concepts.
        
        Returns:
            Tuple of (anthroposophy_embeddings, vedanta_embeddings)
        """
        if not self.model:
            logger.error("Sentence transformer model not available")
            return {}, {}
        
        logger.info("Computing concept embeddings...")
        
        # Compute embeddings for Anthroposophy concepts
        anthro_embeddings = {}
        for concept, description in self.concept_pairs['anthroposophy'].items():
            text = f"{concept}: {description}"
            embedding = self.model.encode(text)
            anthro_embeddings[concept] = embedding
        
        # Compute embeddings for Vedanta concepts
        vedanta_embeddings = {}
        for concept, description in self.concept_pairs['vedanta'].items():
            text = f"{concept}: {description}"
            embedding = self.model.encode(text)
            vedanta_embeddings[concept] = embedding
        
        return anthro_embeddings, vedanta_embeddings
    
    def find_parallel_concepts(self, top_n: int = 5) -> List[Dict]:
        """
        Find most similar concepts between traditions.
        
        Args:
            top_n: Number of top matches per concept
            
        Returns:
            List of concept mapping dictionaries
        """
        if not self.model:
            logger.error("Cannot find parallel concepts without sentence transformer")
            return []
        
        logger.info("Finding parallel concepts...")
        
        anthro_embeddings, vedanta_embeddings = self.compute_concept_embeddings()
        
        # Compute all pairwise similarities
        mappings = []
        
        for anthro_concept, anthro_emb in anthro_embeddings.items():
            similarities = []
            
            for vedanta_concept, vedanta_emb in vedanta_embeddings.items():
                similarity = cosine_similarity(
                    anthro_emb.reshape(1, -1),
                    vedanta_emb.reshape(1, -1)
                )[0][0]
                
                similarities.append({
                    'vedanta_concept': vedanta_concept,
                    'vedanta_description': self.concept_pairs['vedanta'][vedanta_concept],
                    'similarity': float(similarity)
                })
            
            # Sort by similarity
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            
            mappings.append({
                'anthroposophy_concept': anthro_concept,
                'anthroposophy_description': self.concept_pairs['anthroposophy'][anthro_concept],
                'top_matches': similarities[:top_n]
            })
        
        return mappings
    
    def create_concept_network(self, mappings: List[Dict], threshold: float = 0.5) -> Dict:
        """
        Create concept network graph data.
        
        Args:
            mappings: Concept mappings from find_parallel_concepts
            threshold: Minimum similarity threshold for edges
            
        Returns:
            Dictionary with nodes and edges for network graph
        """
        nodes = []
        edges = []
        
        # Add Anthroposophy nodes
        for concept in self.concept_pairs['anthroposophy']:
            nodes.append({
                'id': f"anthro_{concept}",
                'label': concept,
                'tradition': 'anthroposophy',
                'description': self.concept_pairs['anthroposophy'][concept]
            })
        
        # Add Vedanta nodes
        for concept in self.concept_pairs['vedanta']:
            nodes.append({
                'id': f"vedanta_{concept}",
                'label': concept,
                'tradition': 'vedanta',
                'description': self.concept_pairs['vedanta'][concept]
            })
        
        # Add edges based on similarity
        for mapping in mappings:
            anthro_concept = mapping['anthroposophy_concept']
            
            for match in mapping['top_matches']:
                if match['similarity'] >= threshold:
                    edges.append({
                        'source': f"anthro_{anthro_concept}",
                        'target': f"vedanta_{match['vedanta_concept']}",
                        'weight': match['similarity']
                    })
        
        return {
            'nodes': nodes,
            'edges': edges
        }
    
    def generate_mapping_report(self, mappings: List[Dict]) -> str:
        """
        Generate human-readable concept mapping report.
        
        Args:
            mappings: Concept mappings
            
        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("CONCEPT MAPPING: ANTHROPOSOPHY ↔ VEDANTA")
        report.append("=" * 80)
        report.append("\nThis analysis identifies potential parallel concepts between the two traditions.")
        report.append("Similarities are computed using semantic embeddings.\n")
        
        for mapping in mappings:
            report.append("\n" + "-" * 80)
            report.append(f"\nANTHROPOSOPHY CONCEPT: {mapping['anthroposophy_concept'].upper()}")
            report.append(f"Description: {mapping['anthroposophy_description']}")
            report.append("\nMost Similar Vedanta Concepts:")
            
            for i, match in enumerate(mapping['top_matches'][:3], 1):
                report.append(f"\n  {i}. {match['vedanta_concept']} (similarity: {match['similarity']:.3f})")
                report.append(f"     {match['vedanta_description']}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)
    
    def analyze_concepts(self, output_dir: Path = Path("results")) -> None:
        """
        Run complete concept analysis.
        
        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.model:
            logger.error("Sentence transformers not available. Install with: pip install sentence-transformers")
            return
        
        # Find parallel concepts
        mappings = self.find_parallel_concepts()
        
        # Save mappings
        with open(output_dir / "concept_mappings.json", 'w') as f:
            json.dump(mappings, f, indent=2)
        
        # Generate and save report
        report = self.generate_mapping_report(mappings)
        with open(output_dir / "concept_mapping_report.txt", 'w') as f:
            f.write(report)
        
        # Create network data
        network = self.create_concept_network(mappings)
        with open(output_dir / "concept_network.json", 'w') as f:
            json.dump(network, f, indent=2)
        
        logger.info(f"Concept analysis complete. Results saved to: {output_dir}")
        print(report)


def main():
    """Main execution for concept mapping."""
    mapper = ConceptMapper()
    mapper.analyze_concepts()


if __name__ == "__main__":
    main()
