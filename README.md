# Article Analyzer with FLAN-T5 and Gradio

This repository contains a web application for analyzing and summarizing articles using a large language model (FLAN-T5) from Hugging Face's Transformers library. The app processes articles from a URL or a local text file, provides a summary, extracts key points, and performs critical analysis. The application is powered by [Gradio](https://gradio.app/) for creating the user interface.

## Features

- **Article Text Extraction**: Automatically fetch and extract text from a given URL.
- **Text Chunking**: Splits large articles into manageable chunks to avoid model input length limits.
- **Summarization & Analysis**: Uses the FLAN-T5 model to summarize the text, highlight key points, and provide critical analysis.
- **Real-time Output**: Displays the analysis in real-time with a typewriter-like effect for enhanced user experience.
- **File Support**: Upload local text files for analysis.
- **Gradio Interface**: Simple and interactive web UI built with Gradio.

## Installation

1. Clone this repository:
    ```bash
    git clone https://github.com/your-username/article-analyzer.git
    cd article-analyzer
    ```

2. Create a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Download the required model (`google/flan-t5-large`) from Hugging Face:
    ```bash
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
    ```

## How to Run

1. Run the Gradio app:
    ```bash
    python app.py
    ```

2. Access the web interface at `http://localhost:7860` in your browser.

## Usage

1. **Analyze URL**: Provide the URL of an article you want to analyze. Click the "Analyze URL" button to generate the summary and analysis.
2. **Analyze Local File**: Upload a local text file. Click the "Analyze file" button to generate the summary and analysis.
3. The output will be displayed with real-time writing animation.

## Project Structure

- `app.py`: The main application file that includes the article extraction, chunking, analysis, and Gradio UI.
- `requirements.txt`: A list of dependencies required to run the project.
- `README.md`: Documentation for the project.

## How It Works

1. **Language Model Initialization**: The app uses `google/flan-t5-large`, a powerful text generation model from Hugging Face's library. The model is wrapped with a `HuggingFacePipeline` to simplify interaction.
2. **Article Text Extraction**: If the input is a URL, the article's content is fetched using `requests` and parsed using `BeautifulSoup`.
3. **Text Chunking**: Articles are split into chunks of a specified length to prevent exceeding the model's input limit.
4. **Summarization**: Each chunk is processed through a Langchain `LLMChain`, which formats the article using a predefined prompt template to generate a summary, key points, and a critical analysis.
5. **Real-time Display**: The generated output is displayed with a typewriter effect using Gradio's `yield` functionality for a dynamic user experience.

## Dependencies

- `gradio`
- `langchain`
- `transformers`
- `requests`
- `beautifulsoup4`

Install them by running:
```bash
pip install gradio langchain transformers requests beautifulsoup4
```

## Limitations

While this tool provides useful analysis, it's important to note that the AI model may sometimes generate inaccurate or hallucinated content. Users should not rely solely on this tool for critical decisions and should verify important information from reliable sources.