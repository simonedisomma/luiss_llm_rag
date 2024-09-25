from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("Starting the application")

# Prepare
documents = [
    ("Password Reset", """
    Password reset is a crucial security feature that allows users to regain access to their accounts when they forget their passwords. The process typically involves several steps to ensure the security and authenticity of the user.

    Common methods for password reset include:
    1. Email verification: A reset link is sent to the user's registered email address.
    2. Security questions: Users answer pre-set questions to verify their identity.
    3. Two-factor authentication: A code is sent to the user's phone or email as an additional security measure.

    Best practices for password reset include using strong encryption, implementing account lockout policies after multiple failed attempts, and encouraging users to create strong, unique passwords.
    """),
    ("Microsoft Excel", """
    Microsoft Excel is a powerful spreadsheet application developed by Microsoft. It's widely used in business and academic settings for data analysis, financial modeling, and various computational tasks.

    Key features of Excel include:
    1. Data organization: Cells, rows, and columns for structured data entry.
    2. Formulas and functions: Built-in mathematical, statistical, and logical operations.
    3. Charts and graphs: Visual representation of data.
    4. Pivot tables: Dynamic summarization and analysis of large datasets.
    5. Macros and VBA: Automation of repetitive tasks through programming.
     
    Excel is good for data analysis and financial modeling.
    
    For problem about excel call Simone Di Somma

    Excel's versatility makes it an essential tool for tasks ranging from simple budget tracking to complex data analysis and reporting in various industries.
    """)
]

logging.info("Documents loaded")

def split_text(text: str, chunk_size: int = 200) -> List[str]:
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

chunks = []
for title, content in documents:
    chunks.extend(split_text(content))

logging.info(f"Text split into {len(chunks)} chunks")

# Embedd
from torch.utils.data import DataLoader

model = SentenceTransformer('paraphrase-MiniLM-L3-v2')  # Smaller, faster model
logging.info("SentenceTransformer model loaded")

def batch_encode(texts, batch_size=32):
    dataloader = DataLoader(texts, batch_size=batch_size)
    embeddings = []
    for batch in dataloader:
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.append(batch_embeddings)
    return np.concatenate(embeddings)

embeddings = batch_encode(chunks)
logging.info(f"Embeddings created with shape {embeddings.shape}")

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings.astype('float32'))
logging.info("FAISS index created")

import tempfile
import os

def create_and_save_index(embeddings):
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings.astype('float32'))
    _, temp_path = tempfile.mkstemp()
    faiss.write_index(index, temp_path)
    logging.info(f"FAISS index saved to {temp_path}")
    return temp_path

index_path = create_and_save_index(embeddings)

# Retrieve
def retrieve_relevant_chunks(query: str, top_k: int = 3) -> List[str]:
    index = faiss.read_index(index_path)
    query_vector = model.encode([query])
    distances, indices = index.search(query_vector.astype('float32'), top_k)
    logging.info(f"Retrieved {top_k} relevant chunks for query: {query}")
    return [chunks[i] for i in indices[0]]

# Generate
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
logging.info(f"Using device: {device}")

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# Load the model
qwen_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
logging.info("Qwen model loaded")

tokenizer = AutoTokenizer.from_pretrained(model_name)
logging.info("Tokenizer loaded")

def generate_answer(query: str, context: str) -> str:
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    messages = [
        {"role": "system", "content": "You are Luiss assistant, created by Simone Di Somma. You are a helpful assistant. answer mainly using paraphrasing the content below "},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(qwen_model.device)

    logging.info("Generating answer")
    with torch.no_grad():
        generated_ids = qwen_model.generate(
            **model_inputs,
            max_new_tokens=512
        )

    # Remove the input tokens to get only the generated part
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    logging.info("Answer generated")
    return response.strip()

# Clear cache before model loading
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
    logging.info("MPS cache cleared before model loading")
elif torch.cuda.is_available():
    torch.cuda.empty_cache()
    logging.info("CUDA cache cleared before model loading")

def rag_query(query: str) -> str:
    logging.info(f"Processing RAG query: {query}")
    relevant_chunks = retrieve_relevant_chunks(query)
    logging.info(f"Retrieved chunks: {relevant_chunks}")
    context = " ".join(relevant_chunks)
    answer = generate_answer(query, context)
    logging.info("RAG query processed")
    return answer

query = "What is the main feature of Microsoft Excel?"
logging.info(f"Executing query: {query}")
result = rag_query(query)
print(result)
logging.info("Query execution completed")