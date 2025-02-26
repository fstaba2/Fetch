 
# Sentence Transformer Implementation: This model should be able to encode input sentences into fixed-length embeddings.
# Author: Farzaneh Tabataba
# Description: Sentence Transformer is a deep learning model to convert sentences into fixed-length vector embeddings. Unlike traditional RNN 
#              models, that feed the sentence word-by-word to the model, sentence transformers consider the whole sentence at once. We can use
#              load the pre-trained models for this purpose and if we have enough data, we can fine-tune the model on our own dataset.
#              The most common framework for this purpose is the Hugging Face Transformers library.
#              There are various pre-trained models trained based on different transformer architectures such as Bert, RoBERTa, T5, etc.
#              In this example, we used two pre-trained models, paraphrase-MiniLM-L6-v2 (smaller model) and paraphrase-mpnet-base-v2 (more accurate one)
#              , to encode the input sentences 
#              The required libraries for this implementation are listed in the requirements.txt file and installed under venv.
#              
#               This script loads the Sentence Transformer model and encodes the input sentences into fixed-length embeddings.
#               The embeddings are then saved in a csv files.
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd


# Load the pretrained models
model1 = SentenceTransformer('paraphrase-MiniLM-L6-v2') # Small and efficient for paraphrase tasks
model2 = SentenceTransformer('paraphrase-mpnet-base-v2') # Higher accuracy for paraphrase identification

phrases = ["King is great.", "Queen is his wife.", "I love Machine learning."]

vector_embeddings = model1.encode(phrases)

print(vector_embeddings.shape)
print(vector_embeddings)

df1 = pd.DataFrame(vector_embeddings)
df1.to_csv('embeddings1.csv', index=False)

vector_embeddings = model2.encode(phrases)
print(vector_embeddings.shape)
print(vector_embeddings)

df2 = pd.DataFrame(vector_embeddings)
df2.to_csv('embeddings2.csv', index=False)
