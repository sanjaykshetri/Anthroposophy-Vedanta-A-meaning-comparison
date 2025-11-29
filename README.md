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
- **Text Collection**: Automated scraping from Project Gutenberg and online archives
- **Reference Detection**: Identify mentions of Vedanta, Bhagavad Gita, Upanishads in Anthroposophical texts
- **Semantic Similarity**: Compare conceptual parallels using TF-IDF and NLP embeddings
- **Topic Modeling**: Discover shared and distinct themes using LDA
- **Statistical Analysis**: Quantitative comparison of vocabulary, concepts, and ideas
- **Comprehensive Visualizations**: 
  - Word clouds for vocabulary analysis
  - Co-occurrence networks showing term relationships
  - Similarity heatmaps and distribution charts
  - Category breakdowns and comparative statistics
  - Publication-ready exports (300 DPI)

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

### Quick Start
```bash
# Clone the repository
git clone https://github.com/sanjaykshetri/Anthroposophy-Vedanta-A-meaning-comparison
cd Anthroposophy-Vedanta-A-meaning-comparison

# Install dependencies
pip install -r requirements.txt

# Download required NLTK data
python -m nltk.downloader punkt stopwords wordnet

# Download spaCy model
python -m spacy download en_core_web_sm

# Collect texts from Project Gutenberg
python expand_corpus.py

# Detect Vedanta references
python src/reference_detection/detect_references.py

# Run interactive analysis notebook
jupyter notebook notebooks/analysis_notebook.ipynb
```

### Analysis Notebook Features
The Jupyter notebook (`notebooks/analysis_notebook.ipynb`) includes:
1. **Reference Detection** - Find and visualize Vedanta mentions
2. **Corpus Statistics** - Compare text sizes and characteristics
3. **Similarity Analysis** - TF-IDF and semantic comparison
4. **Distinctive Terms** - Identify tradition-specific vocabulary
5. **Topic Modeling** - Discover latent themes (LDA)
6. **Word Clouds** - Visual vocabulary comparison
7. **Co-occurrence Networks** - Term relationship graphs
8. **Category Analysis** - Breakdown by concept type
9. **Export Functions** - Save high-res figures for publications

All visualizations are interactive and can be exported at 300 DPI for academic papers.

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

## Current Results

### Corpus Status (Updated: Nov 2025)
- **6 Anthroposophy texts** (~500KB): Including Steiner's major works
- **8 Vedanta texts** (~600KB): Bhagavad Gita translations, Upanishads, Vivekananda's Yogas
- **Total: ~1.1MB** of philosophical text under analysis

### Key Findings
- **26 direct references** to Vedanta concepts found in Anthroposophy texts (85% increase from baseline)
- **17 unique Vedanta terms** identified across corpus
- **Most frequent terms**: maya (4), karma (3), self-knowledge (3), gita (2), vedas (2)
- **Most influenced text**: Steiner's "Spiritual Guidance" (12 Vedanta references)
- Steiner explicitly acknowledges Vedanta as having "recognized similar truths"
- Strong conceptual parallels in consciousness, reincarnation, and spiritual development

### Available Visualizations
- Reference frequency charts and text-by-text distribution
- Word clouds comparing vocabulary patterns
- Term co-occurrence networks
- Corpus statistics and size comparisons
- Category breakdowns (Texts, Concepts, Practices, Figures)
- Similarity heatmaps (when full analysis is run)

See `results/vedanta_references_report.txt` for detailed findings and `notebooks/analysis_notebook.ipynb` for interactive exploration.

---

## 🚀 Future Directions & Next Steps

### Immediate Priorities
1. **Expand Text Corpus**
   - Collect 50-100 texts per tradition
   - Add complete Steiner lecture series from rsarchive.org
   - Include full Upanishads collection and Shankara's commentaries
   - Integrate Vivekananda's complete works

2. **Enhanced Analysis**
   - Implement BERT/Sentence-BERT for deeper semantic analysis
   - Track temporal evolution of Vedanta references in Steiner's works
   - Build cross-reference network graphs
   - Statistical hypothesis testing for influence patterns

### Research Streams

#### 🔬 **Stream A: Advanced NLP**
- Modern transformer models (BERT, GPT embeddings)
- Cross-lingual analysis (Sanskrit ↔ German ↔ English)
- Fine-tune models on philosophical texts
- Automated concept extraction

#### 📊 **Stream B: Historical Analysis**
- Timeline visualization of concept evolution
- Track which Vedanta translations Steiner accessed
- Chronological mapping of Eastern influence
- Citation network analysis

#### 🗺️ **Stream C: Comparative Philosophy**
- Expand to Buddhism, Sufism, Christian Mysticism, Kabbalah
- Build multi-tradition similarity matrix
- Identify unique vs. universal concepts
- Create comprehensive "spiritual philosophy knowledge graph"

#### 🤖 **Stream D: Machine Learning**
- Train classifiers to distinguish traditions
- Predict tradition from text passages
- Anomaly detection for hybrid/borrowed concepts
- Feature importance analysis

#### 🌐 **Stream E: Interactive Tools**
- Streamlit/Flask web dashboard
- "Find Similar Passages" search engine
- Real-time text analysis API
- Chrome extension for reading assistance

### Academic Opportunities

#### 📝 **Publication Targets**
- **Digital Humanities Quarterly** - Computational methodology
- **Journal of Religion and Computation** - NLP in religious studies
- **Comparative Philosophy** - Cross-tradition analysis
- **Anthroposophy journals** - Findings on Eastern influence

#### 🎓 **Conference Presentations**
- Digital Humanities conferences
- American Academy of Religion
- Anthroposophy research forums
- Religious studies symposia

#### 📚 **Potential Papers**
1. "Computational Analysis of Vedanta Influence in Rudolf Steiner's Anthroposophy"
2. "Digital Humanities Approaches to Comparative Spiritual Philosophy"
3. "Network Analysis of Concept Borrowing in Early 20th Century Esotericism"

### Community Engagement

- **Open Source Contribution**: Expand corpus collaboratively
- **Workshops**: Teach digital humanities methods to philosophy researchers
- **Educational Tools**: Create study guides and interactive learning modules
- **Blog Series**: Share findings on Medium/Substack

---

## 🎯 Quick Win Projects (1-2 Weeks Each)

1. **Visualization Dashboard**: Interactive Streamlit app for exploring results
2. **Concept Network Graph**: NetworkX visualization of term relationships
3. **Medium Article**: "I Used NLP to Prove Steiner's Eastern Influences"
4. **API Endpoint**: Text similarity service for researchers
5. **Jupyter Tutorial**: "Computational Philosophy for Beginners"

---

## Contributing

This is a unique, first-of-its-kind project in computational comparative philosophy. Contributions welcome in:
- Text collection and curation
- Analysis methodology improvements
- Visualization enhancements
- Documentation and tutorials
- Cross-tradition expansion

See `GETTING_STARTED.md` for development setup.

---

## License
MIT License

## Acknowledgments
- Rudolf Steiner Archive
- Sacred Texts Online
- Project Gutenberg
- Vedanta Digital Library

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{anthroposophy_vedanta_comparison,
  author = {Sanjay Kshetri},
  title = {Anthroposophy-Vedanta: A Computational Linguistics Comparison},
  year = {2025},
  url = {https://github.com/sanjaykshetri/Anthroposophy-Vedanta-A-meaning-comparison}
}
```
