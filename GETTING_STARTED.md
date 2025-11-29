# Getting Started Guide

## Installation

1. **Clone the repository**
```bash
cd /workspaces/Anthroposophy-Vedanta-A-meaning-comparison
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download required NLP models**
```bash
# Download NLTK data
python -m nltk.downloader punkt stopwords wordnet

# Download spaCy model
python -m spacy download en_core_web_sm
```

## Quick Start

### 1. Collect Texts

Run the data collection script to gather texts from online sources:

```bash
cd src/data_collection
python collect_texts.py --source all
```

This will collect:
- **Anthroposophy**: Rudolf Steiner's works from Project Gutenberg
- **Vedanta**: Bhagavad Gita, Upanishads from various sources

### 2. Run the Complete Analysis Pipeline

```bash
cd src
python run_analysis.py --all
```

Or run individual steps:

```bash
# Preprocess texts
python run_analysis.py --preprocess

# Detect Vedanta references
python run_analysis.py --references

# Compute similarities
python run_analysis.py --similarity

# Map concepts
python run_analysis.py --concepts
```

### 3. Explore Results

Results are saved in the `results/` directory:

- `vedanta_references_report.txt` - Frequency of Vedanta mentions
- `similarity_analysis.json` - Text similarity metrics
- `distinctive_terms.json` - Characteristic vocabulary
- `topic_modeling.json` - Discovered topics
- `concept_mappings.json` - Parallel concepts
- `similarity_heatmap.png` - Visual similarity matrix

### 4. Interactive Analysis

Use the Jupyter notebook for interactive exploration:

```bash
cd notebooks
jupyter notebook analysis_notebook.ipynb
```

## Individual Module Usage

### Reference Detection

```python
from src.reference_detection.detect_references import ReferenceDetector
from pathlib import Path

detector = ReferenceDetector()
analysis = detector.analyze_corpus(
    corpus_dir=Path("data/anthroposophy"),
    output_path=Path("results/references.json")
)
report = detector.generate_report(analysis)
print(report)
```

### Similarity Analysis

```python
from src.analysis.similarity_analysis import SimilarityAnalyzer

analyzer = SimilarityAnalyzer()
analyzer.load_texts()
similarity_results = analyzer.compute_tfidf_similarity()
print(f"Average similarity: {similarity_results['average_similarity']:.4f}")
```

### Concept Mapping

```python
from src.analysis.concept_mapping import ConceptMapper

mapper = ConceptMapper()
mappings = mapper.find_parallel_concepts(top_n=5)
report = mapper.generate_mapping_report(mappings)
print(report)
```

## Research Questions Addressed

1. **How often are Vedanta concepts mentioned?**
   - Run reference detection to find frequency of terms like "Vedanta", "Bhagavad Gita", "Atman", etc.
   
2. **How similar are the texts semantically?**
   - Use similarity analysis to compute cosine similarity between documents
   
3. **What are the distinctive concepts?**
   - Extract distinctive terms to see what makes each tradition unique
   
4. **What are the parallel concepts?**
   - Use concept mapping to find semantic equivalents (e.g., "ego" vs "atman")
   
5. **What topics are shared vs. distinct?**
   - Topic modeling reveals thematic overlap and divergence

## Troubleshooting

### No texts found error
- Ensure data collection has been run first
- Check that files exist in `data/anthroposophy/` and `data/vedanta/`

### Memory errors
- Reduce `max_features` in config.yaml
- Process fewer texts at once
- Use a machine with more RAM

### Missing dependencies
```bash
pip install -r requirements.txt --upgrade
```

### SpaCy model not found
```bash
python -m spacy download en_core_web_sm
```

### NLTK data not found
```bash
python -m nltk.downloader all
```

## Next Steps

1. **Expand the corpus**: Add more texts from additional sources
2. **Refine concept pairs**: Update concept_mapping.py with more nuanced mappings
3. **Deep analysis**: Use BERT or other transformers for semantic analysis
4. **Visualization**: Create network graphs of concept relationships
5. **Statistical testing**: Add significance tests for differences

## Support & Contributing

For issues or contributions, please check the project repository.

Happy researching! 🔍📚
