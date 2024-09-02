import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
import torch
import base64
import tempfile
import os

# Model and tokenizer loading
checkpoint = "LaMini-Flan-T5-248M"
tokenizer = T5Tokenizer.from_pretrained(checkpoint)
base_model = T5ForConditionalGeneration.from_pretrained(checkpoint, device_map='auto', torch_dtype=torch.float32)

# File loader and preprocessing
def file_preprocessing(uploaded_file):
    # Use a temporary file to handle the PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)

    final_texts = ""
    for text in texts:
        final_texts += text.page_content

    # Clean up temporary file
    os.remove(temp_file_path)
    
    return final_texts

# LLM pipeline
def llm_pipeline(uploaded_file):
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=500,
        min_length=50
    )
    input_text = file_preprocessing(uploaded_file)
    # Chunk the text for summarization to handle large texts
    chunks = [input_text[i:i + 1024] for i in range(0, len(input_text), 1024)]
    summaries = []
    
    for chunk in chunks:
        result = pipe_sum(chunk)
        summaries.append(result[0]['summary_text'])
    
    # Combine summaries into a final summary
    final_summary = " ".join(summaries)
    return final_summary

@st.cache_data
# Function to display the PDF of a given file 
def displayPDF(file):
    # Opening file from file-like object
    base64_pdf = base64.b64encode(file.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

# Streamlit code 
st.set_page_config(layout="wide")

def main():
    st.title("Document Summarization App using Language Model")

    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    if uploaded_file is not None:
        if st.button("Summarize"):
            col1, col2 = st.columns(2)

            with col1:
                st.info("Uploaded File")
                displayPDF(uploaded_file)  # Display the uploaded file

            with col2:
                # Use the file-like object directly
                summary = llm_pipeline(uploaded_file)  
                st.info("Summarization Complete")
                st.success(summary)

if __name__ == "__main__":
    main()
