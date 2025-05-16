import nltk
nltk.download('punkt_tab')

import os

nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_dir)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_dir)

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

from transformers import pipeline
import gradio as gr

# Extractive Summarization using Sumy (LSA)
def extractive_summary(text, num_sentences=3):
    try:
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        total_sentences = len(parser.document.sentences)
        if total_sentences == 0:
            return "Text too short or improperly formatted for summarization."

        # Adjust number of sentences to summarize
        num_sentences = min(num_sentences, total_sentences)

        summarizer = LsaSummarizer()
        summary = summarizer(parser.document, num_sentences)

        return " ".join(str(sentence) for sentence in summary)
    except Exception as e:
        return f"Error in extractive summarization: {str(e)}"


# Abstractive Summarization using BART
abstractive_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def abstractive_summary(text):
    try:
        summary = abstractive_summarizer(text, max_length=130, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error in abstractive summarization: {str(e)}"

# Combined summarization function
def summarize(text, method):
    if method == "Extractive":
        summary = extractive_summary(text)
    else:
        summary = abstractive_summary(text)
    return text, summary

# Gradio Interface
iface = gr.Interface(
    fn=summarize,
    inputs=[
        gr.Textbox(lines=15, label="Enter Text Here"),
        gr.Radio(choices=["Extractive", "Abstractive"], label="Summarization Method")
    ],
    outputs=[
        gr.Textbox(label="Original Text"),
        gr.Textbox(label="Summarized Text")
    ],
    title="Text Summarization Tool",
    description="Choose between Extractive (Sumy) or Abstractive (BART) summarization methods."
)

iface.launch(inbrowser=True, share=False)           




