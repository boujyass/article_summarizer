import gradio as gr
from langchain import LLMChain, PromptTemplate
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import requests
from bs4 import BeautifulSoup
import time

# Initialize the language model (FLAN-T5)
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=512,
    temperature=0.1,
    do_sample=True,  # Enable sampling
)

llm = HuggingFacePipeline(pipeline=pipe)

template = """
Analyze the following article excerpt and provide a concise summary, key points, and a critical analysis:

Article excerpt: {article}

Summary:
Key Points:
Critical Analysis:

"""

prompt = PromptTemplate(
    input_variables=["article"],
    template=template
)

article_analyzer = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True
)

def extract_article_text(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')
    article_text = ' '.join([p.text for p in paragraphs])
    return article_text

def chunk_text(text, chunk_size=3000):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def analyze_article(article_source, progress=gr.Progress()):
    if article_source.startswith('http'):
        article_text = extract_article_text(article_source)
    else:
        article_text = article_source

    chunks = chunk_text(article_text)
    results = []

    for i, chunk in enumerate(chunks):
        result = article_analyzer.predict(article=chunk)
        results.append(result)
        progress((i + 1) / len(chunks))

    final_result = "\n\n".join(results)
    
    # Simulate a writing animation
    words = final_result.split()
    output = ""
    for i, word in enumerate(words):
        output += word + " "
        progress((i + 1) / len(words))
        time.sleep(0.05)  # Adjust to control writing speed
        yield output

def process_file(file):
    return file.read().decode('utf-8')

with gr.Blocks() as demo:
    gr.Markdown("# Article Analyzer with Animation")
    gr.Markdown("**Disclaimer**: This AI-generated analysis may contain inaccuracies. Please verify important information.")
    
    
    with gr.Tab("URL"):
        url_input = gr.Textbox(label="Article URL")
        url_button = gr.Button("Analyze URL")
    
    with gr.Tab("Local File"):
        file_input = gr.File(label="Import a text file")
        file_button = gr.Button("Analyze file")
    
    output = gr.Textbox(label="Summary and Analysis", lines=10)
    
    url_button.click(analyze_article, inputs=url_input, outputs=output)
    file_button.click(analyze_article, inputs=file_input, outputs=output)

demo.launch()