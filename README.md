# RAG Project for Football Videos

A Retrieval-Augmented Generation (RAG) system for processing football videos downloaded via yt-dlp, converting them to text using Whisper, and performing intelligent semantic search with LLM-powered responses.

## Overview

This project creates an intelligent football knowledge base by:

1. **Downloading** football videos from YouTube using yt-dlp
2. **Transcribing** audio to text using OpenAI's Whisper large-v2 model
3. **Chunking** transcriptions into semantically meaningful segments
4. **Embedding** text chunks using Ollama's bge-m3 model
5. **Searching** relevant content using cosine similarity
6. **Generating** natural language responses with DeepSeek-R1 LLM

## Features

- **🎥 Video Processing**: Automated download and processing of football videos
- **🎤 Speech-to-Text**: High-accuracy transcription using Whisper large-v2
- **📝 Intelligent Chunking**: Context-aware text segmentation
- **🔍 Semantic Search**: Vector-based similarity search with embeddings
- **🤖 LLM Integration**: Natural language responses with video references
- **⚡ Caching**: Optimized performance with joblib embedding cache
- **💾 Local Processing**: Complete privacy - all processing done locally

## Requirements

```bash
openai-whisper
requests
pandas
scikit-learn
numpy
joblib
```

## Setup

1. **Install Python dependencies:**

```bash
pip install openai-whisper requests pandas scikit-learn numpy joblib
```

2. **Install and configure Ollama:**

```bash
# Install Ollama from https://ollama.ai
# Pull required models
ollama pull bge-m3
ollama pull deepseek-r1
```

3. **Start Ollama server:**

```bash
ollama serve
```

Ensure Ollama is running on `http://localhost:11434`

## Project Structure

```
rag_project/
├── stt.py                  # Whisper model loader for speech-to-text
├── read_chunks.py          # Embedding generator for video chunks
├── query_processor.py      # Main query processing and LLM integration
├── input_output.ipynb      # Interactive notebook for testing queries
├── data.json              # Video chunks with metadata and embeddings
├── embeddings.joblib      # Cached embeddings for fast loading
├── prompt.txt             # Last generated prompt for debugging
└── content/               # Downloaded video content directory
```

## Usage

### 1. Initial Setup - Load Whisper Model

```bash
python stt.py
```

This downloads and loads the Whisper large-v2 model (~2.87 GB on first run).

### 2. Process Video Chunks and Generate Embeddings

```bash
python read_chunks.py
```

This processes your video transcription chunks and generates embeddings, saving them to `data.json` and `embeddings.joblib`.

### 3. Query the Knowledge Base

**Option A: Using Python Script**

```bash
python query_processor.py
```

**Option B: Using Jupyter Notebook**

Open `input_output.ipynb` and run cells interactively to test different queries.

**Option C: Programmatic Usage**

```python
from query_processor import process_query

# Ask a question
response = process_query("Tell me about Marco van Basten")
print(response)
```

### Example Queries

- "Tell me about Marco van Basten"
- "Who is Lionel Messi?"
- "What makes Messi different from Ronaldo?"
- "Best national football teams in history"

## Files

- `stt.py` - Loads the Whisper large-v2 model for speech-to-text conversion
- `read_chunks.py` - Processes text chunks and generates embeddings
- `query_processor.py` - Main query processing with LLM integration
- `input_output.ipynb` - Interactive notebook for testing queries
- `data.json` - Stores video chunks with embeddings and metadata
- `embeddings.joblib` - Cached embeddings for fast loading
- `prompt.txt` - Last generated prompt for debugging

## How It Works

1. **Video Download**: Videos are downloaded from YouTube using yt-dlp
2. **Speech Recognition**: Whisper transcribes audio to text with timestamps
3. **Text Chunking**: Transcripts are segmented into contextual chunks
4. **Embedding Generation**: Each chunk is converted to a vector using bge-m3
5. **Query Processing**: User queries are embedded and compared to chunk embeddings
6. **Semantic Search**: Cosine similarity identifies the most relevant chunks
7. **LLM Response**: DeepSeek-R1 generates natural language answers with video references

## Notes

- The Whisper large-v2 model requires ~2.87 GB download on first run
- Embeddings are generated locally using Ollama's bge-m3 model
- All processing is done locally without external API calls
- Cached embeddings in `embeddings.joblib` enable instant loading
- Responses include video titles and timestamps for easy reference
- Works completely offline after initial model downloads

## Example Output

```
Here's information about Marco van Basten based on Markaroni's video content:

Marco Van Basten was a phenomenal footballer known for his incredible skill and finishing. His impact began young: "Video: How good was marco vanbasten at 0:53" highlights that he was already ahead of others at age 16, while another snippet states, "Video: How good was marco vanbasten at 2:49" confirms this early dominance.

He had a legendary career with AC Milan, scoring crucial goals and becoming one of the greatest striikers ever. For instance, "Video: How good was marco vanbasten at 3:05" details his teaching moment for young talents, and another says, "Video: How good was marco vanbasten at 7:16" notes him as a top scorer during significant periods.

His influence extended globally too. Look up specific goals or anecdotes in the following videos:      
*   Video: How good was marco vanbasten at 5:08 - Discusses his famous goal and impact.
*   Video: How good was marco vanbasten at 13:26 - Covers how he influenced European markets.

```


