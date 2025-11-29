# RAG Project for Football Videos

A Retrieval-Augmented Generation (RAG) system for processing football videos downloaded via yt-dlp, converting them to text using Whisper, and performing semantic search using embeddings.

## Overview

This project processes football videos locally by:

1. Downloading videos using yt-dlp
2. Converting audio to text using OpenAI's Whisper model (large-v2)
3. Chunking the transcribed text
4. Creating embeddings using Ollama's bge-m3 model
5. Performing semantic search using cosine similarity in pandas DataFrames

## Features

- **Video Processing**: Downloads and processes football videos from YouTube
- **Speech-to-Text**: Uses Whisper large-v2 model for accurate transcription
- **Text Chunking**: Segments transcriptions into manageable chunks
- **Embeddings**: Generates embeddings using Ollama's bge-m3 model
- **Semantic Search**: Finds relevant content using cosine similarity

## Requirements

```
openai-whisper
requests
pandas
scikit-learn
numpy
```

## Setup

1. Install Python dependencies:

```bash
pip install openai-whisper requests pandas scikit-learn numpy
```

2. Install Ollama and pull the bge-m3 model:

```bash
ollama pull bge-m3
```

3. Ensure Ollama is running locally on port 11434

## Files

- `stt.py` - Loads the Whisper large-v2 model for speech-to-text conversion
- `read_chunks.py` - Processes text chunks and generates embeddings
- `data.json` - Stores video chunks with embeddings and metadata
- `Untitled-1.ipynb` - Jupyter notebook for semantic search demonstrations

## Usage

1. Load the Whisper model:

```bash
python stt.py
```

2. Process chunks and generate embeddings:

```bash
python read_chunks.py
```

3. Use the Jupyter notebook for semantic search queries

## Project Structure

```
rag_project/
├── stt.py              # Whisper model loader
├── read_chunks.py      # Embedding generator
├── data.json           # Video chunks with embeddings
└── content/            # Downloaded video content
```

## Notes

- The Whisper large-v2 model requires ~2.87 GB download on first run
- Embeddings are generated locally using Ollama
- All processing is done locally without external API calls

## License

MIT
