# Data Collection Module

This module handles the collection of texts from various online sources for both Anthroposophy and Vedanta traditions.

## Sources

### Anthroposophy
- **Project Gutenberg**: Rudolf Steiner's major works
- **Rudolf Steiner Archive** (rsarchive.org)
- **Sacred Texts** (sacred-texts.com)

### Vedanta
- **Project Gutenberg**: Bhagavad Gita translations
- **Sacred Texts**: Upanishads, Brahma Sutras
- **Internet Archive**: Shankara's commentaries

## Usage

```bash
# Collect all texts
python collect_texts.py --source all

# Collect only Anthroposophy texts
python collect_texts.py --source anthroposophy

# Collect only Vedanta texts
python collect_texts.py --source vedanta
```

## Output Structure

```
data/
├── anthroposophy/
│   ├── metadata.json
│   ├── steiner_*.txt
│   └── ...
└── vedanta/
    ├── metadata.json
    ├── gita_*.txt
    ├── upanishad_*.txt
    └── ...
```

## Notes

- Scripts include rate limiting to be respectful to servers
- Metadata is saved for each collection
- Individual text files are saved for easy processing
- Some URLs may need updating based on source availability
