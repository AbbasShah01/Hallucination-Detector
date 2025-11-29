# Setup Guide

## Quick Setup

### 1. Clone Repository

```bash
git clone https://github.com/AbbasShah01/Hallucination-Detector.git
cd Hallucination-Detector
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install spaCy Model (Optional)

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

### 5. Configure API Keys (Optional)

For agentic verification with OpenAI:
```bash
export OPENAI_API_KEY="your-key-here"
```

For Anthropic:
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
```

### 6. Run Preprocessing

```bash
python src/preprocess_halueval.py
```

### 7. Run Pipeline

```bash
python src/master_pipeline.py
```

## Verification

Check that everything is installed:

```bash
python -c "import torch; import transformers; import spacy; print('All dependencies installed!')"
```

## Troubleshooting

### Issue: spaCy model not found

**Solution:**
```bash
python -m spacy download en_core_web_sm
```

### Issue: CUDA/GPU not available

**Solution:** The system will automatically use CPU. For GPU support:
- Install CUDA-enabled PyTorch
- Ensure CUDA drivers are installed

### Issue: Import errors

**Solution:**
```bash
pip install --upgrade -r requirements.txt
```

### Issue: Out of memory during training

**Solution:**
- Reduce batch size in `config.json`
- Use gradient accumulation
- Process data in smaller chunks

## System Requirements

- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 5GB free space
- **GPU**: Optional but recommended for faster training

## Next Steps

1. Review `config.json` for customization
2. Check `MASTER_PIPELINE_README.md` for detailed usage
3. Explore `docs/SYSTEM_ARCHITECTURE.md` for architecture details

