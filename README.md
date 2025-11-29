# Anthroposophy-Vedanta: A Computational Linguistics Comparison

## Overview
This project uses computational linguistics and NLP techniques to analyze and compare two major spiritual philosophical traditions:
- **Anthroposophy**: Rudolf Steiner's spiritual philosophy and teachings
- **Vedanta**: Ancient Indian philosophy including Upanishads, Bhagavad Gita, and Adi Shankara's teachings

## Research Questions
1. How much did Rudolf Steiner borrow ideas from ancient Vedanta texts?
2. How often do Steiner and other anthroposophists mention or refer to 'Vedanta', Bhagavad Gita, etc.?
3. In what ways are these two schools similar and distinct?
4. What are the semantic parallels between key concepts in both traditions?

## Project Structure
```
├── data/                          # Text corpora
│   ├── anthroposophy/             # Steiner's works and anthroposophical texts
│   ├── vedanta/                   # Vedanta texts (Upanishads, Gita, Shankara)
│   └── processed/                 # Cleaned and preprocessed texts
├── src/                           # Source code
│   ├── data_collection/           # Web scraping and text collection
│   ├── preprocessing/             # Text cleaning and normalization
│   ├── analysis/                  # Similarity and semantic analysis
│   ├── reference_detection/       # Citation and reference detection
│   └── visualization/             # Results visualization
├── notebooks/                     # Jupyter notebooks for exploration
├── results/                       # Analysis outputs
├── config/                        # Configuration files
└── requirements.txt               # Python dependencies
```

## Features
- **Text Collection**: Automated scraping of texts from online sources
- **Reference Detection**: Identify mentions of Vedanta, Bhagavad Gita, Upanishads in Anthroposophical texts
- **Semantic Similarity**: Compare conceptual parallels using NLP embeddings
- **Topic Modeling**: Discover shared and distinct themes
- **Statistical Analysis**: Quantitative comparison of vocabulary, concepts, and ideas
- **Visualization**: Interactive dashboards and plots

## Methodology
1. **Data Collection**: Gather texts from Project Gutenberg, sacred-texts.com, anthroposophy.org
2. **Preprocessing**: Tokenization, lemmatization, stopword removal
3. **Feature Extraction**: TF-IDF, word embeddings (Word2Vec, BERT)
4. **Similarity Analysis**: Cosine similarity, semantic distance metrics
5. **Named Entity Recognition**: Detect references to texts, concepts, and figures
6. **Topic Modeling**: LDA, BERTopic for theme discovery
7. **Visualization**: Network graphs, heatmaps, word clouds

## Technologies
- Python 3.9+
- NLTK, spaCy for NLP
- Scikit-learn for ML
- Transformers (BERT, Sentence-BERT)
- Beautiful Soup, Scrapy for web scraping
- Pandas, NumPy for data manipulation
- Matplotlib, Seaborn, Plotly for visualization

## Getting Started
```bash
# Install dependencies
pip install -r requirements.txt

# Download required NLTK data
python -m nltk.downloader punkt stopwords wordnet

# Download spaCy model
python -m spacy download en_core_web_sm

# Run data collection
python src/data_collection/collect_texts.py

# Run analysis pipeline
python src/analysis/run_analysis.py
```

## Analysis Components

### 1. Reference Detection
Identify explicit mentions of:
- Terms: "Vedanta", "Bhagavad Gita", "Upanishads", "Brahman", "Atman", "Maya"
- Figures: "Shankara", "Vyasa", "Krishna"
- Concepts specific to Vedanta philosophy

### 2. Semantic Similarity
- Compare key concepts (e.g., "Self" in Anthroposophy vs "Atman" in Vedanta)
- Measure document-level similarity
- Identify parallel passages

### 3. Conceptual Mapping
- Map Anthroposophical terms to Vedantic equivalents
- Analyze conceptual overlaps and divergences

## License
MIT License

## Acknowledgments
- Rudolf Steiner Archive
- Sacred Texts Online
- Project Gutenberg
- Vedanta Digital Library
