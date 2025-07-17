# Deposition Summarizer

An AI-powered tool that transforms lengthy legal depositions into structured, accurate summaries using advanced prompt engineering and topic-based chunking.

## Overview

This project reduces hours of manual deposition review to minutes by:
- Using intelligent topic-based chunking to maintain context across 100+ page documents
- Implementing two-stage summarization (segment analysis → final synthesis)
- Supporting multiple deposition roles (Plaintiff, Defendant, Expert Witness, Fact Witness)
- Achieving 75%+ time savings while maintaining accuracy

## Key Features

- **Smart Topic Detection**: Automatically identifies and segments depositions by legal topics (Personal Info, Employment, Medical History, Incident Details, etc.)
- **Multi-LLM Support**: Works with OpenAI, Anthropic, Google, and OpenRouter APIs
- **Role-Specific Summaries**: Customized output formats for different deponent types
- **Robust Error Handling**: Self-healing pipelines with retry logic and fallback mechanisms
- **Professional Output**: Generates formatted DOCX files ready for legal use

## Development Approach

This tool was developed through an iterative process of prompt engineering and LLM orchestration. Rather than traditional coding, I designed the system by:
1. Identifying friction points in legal workflows
2. Architecting solutions through carefully crafted prompt chains
3. Iteratively refining prompts based on output quality
4. Testing with real deposition transcripts to ensure accuracy

## Technical Architecture

- **app.py**: Main Streamlit application handling UI and orchestration
- **enhanced_topic_chunker_fixed.py**: Advanced chunking algorithm for topic-based segmentation
- **instructions/**: Role-specific prompts and synthesis templates
- **Multi-stage processing**: Context generation → Topic chunking → Segment summarization → Final synthesis

## Usage

1. Upload a deposition transcript (PDF, DOCX, or TXT)
2. Select the deponent's role
3. Configure LLM providers (supports multiple for cost optimization)
4. Generate comprehensive summaries in minutes

## Results

- Reduces 100+ page depositions to accurate summaries in under 10 minutes
- Maintains critical legal details while eliminating redundancy
- Produces attorney-ready work product

## Future Enhancements

- OCR support for scanned depositions
- Batch processing capabilities
- Citation tracking and validation
- Integration with case management systems

---

*Built by Christopher D. Snead - Bridging legal expertise with AI innovation*