# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Streamlit-based Deposition Transcript Summarizer that processes legal deposition transcripts using various LLM providers (OpenAI, Anthropic, Google, OpenRouter). The application uses sophisticated topic-based chunking to intelligently segment depositions while preserving context.

## Key Architecture Components

### Core Files
- **app.py**: Main Streamlit application containing:
  - Multi-provider LLM integration (OpenAI, Anthropic, Google, OpenRouter)
  - File parsing for PDF and DOCX transcripts
  - Two-step summarization process (global context → chunk summaries)
  - Custom instruction support for specialized outlines
  
- **enhanced_topic_chunker_fixed.py**: Sophisticated topic-based chunking implementation that:
  - Uses multi-method topic detection (LLM + pattern-based)
  - Provides quality scoring and segment optimization
  - Includes fallback mechanisms for reliability
  - Tracks processing statistics

### Key Functions in app.py
- `call_llm()`: Unified interface for all LLM providers
- `call_llm_with_retry()`: Adds retry logic with exponential backoff
- `enhanced_chunk_text()`: Primary chunking method using EnhancedTopicBasedChunker
- `chunk_text()`: Fallback chunking using RecursiveCharacterTextSplitter
- `validate_chunks()`: Ensures chunk quality before processing

## Running the Application

### Windows (using Launch.bat):
```bash
Launch.bat
```

### Manual startup:
```bash
# Activate virtual environment
.venv\Scripts\activate.bat  # Windows
source .venv/bin/activate   # Linux/Mac

# Run Streamlit
streamlit run app.py
```

### Development commands:
```bash
# Install dependencies
pip install -r requirements.txt

# Run with specific port
streamlit run app.py --server.port 8502
```

## Configuration

### Environment Variables (.env file required):
```
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
```

### Default Chunking Parameters:
- Target chunk size: 8000 characters
- Min chunk size: 2000 characters  
- Max chunk size: 12000 characters
- Overlap size: 400 characters
- Quality threshold: 0.7

### Output Directory:
- Windows: `~/Documents/Depo_Summaries/`
- Linux/Mac: `~/Depo_Summaries/`
- Logs stored in: `{OUTPUT_DIRECTORY}/logs/`

## Processing Flow

1. **Upload**: PDF or DOCX deposition transcript
2. **Configuration**: Select LLM provider, model, witness type
3. **Custom Instructions**: Add name spellings or special context
4. **Processing**:
   - Text extraction from document
   - Enhanced topic-based chunking (if LLM available)
   - Global context generation
   - Individual chunk summarization
   - Final summary compilation
5. **Output**: DOCX file with structured summary

## Topic Detection Patterns

The enhanced chunker recognizes common deposition topics:
- Personal Information (name, address)
- Employment History
- Medical History (pre-incident)
- The Incident/Accident
- Post-Incident Events
- Medical Treatment
- Current Condition

## Error Handling

- Automatic fallback from topic-based to simple chunking
- Retry logic for transient LLM errors
- DOCX generation fallback to TXT format
- Comprehensive logging for debugging