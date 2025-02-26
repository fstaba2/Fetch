
# Expand the sentence transformer to handle a multi-task learning setting.
# Task A: Sentence Classification, Task B: Sentiment Analysis.
# Author: Farzaneh Tabataba

# Description: In Multi-Task Learning, we train a single model to learn senrence representation and perform multiple tasks. 
# Multi-task learning can benefit from having shared low-level features and shared data between multiple tasks. 
# For NLP tasks, we can use a shared encoder to learn sentence representations and then use different decoders (heads) for different tasks.
# For the two tasks of Sentence Classification and Sentiment Analysis, we can use the same encoder to learn sentence representations 
# and then use two different decoders for output.
# Example of TASK A: Sentence Classification: Two-class classification, of positive and negatiive class that sentence belongs to SPORT or not.
# TASK B: Sentiment analysis is usually three class classification with labels (positive, negative, neutral). 
# For both tasks A, B, we can use a Fully connected lasyer (FC) with softmax activation function as the output layer.
# The loss function for both tasks can be the cross-entropy loss function.
# we assume we have database with labels for all tasks. and we train the model on all data. 
# However, for the task of Named Entity Recogniton (NER), we need the word-level embedding 
# and the output for each word would be one of the classes like (person, company, location, etc).
# So we use a FC layer with softmax activation function for each word in the sentence.
# The loss function is measured for each token with Cross-Entropy loss function. 
# The total  loss function could be the weighted average of the loss functions for all tasks.
# We can train the model on multiple batches of data for each task. 
# The optimization could be done by Gradient Descent or Adam optimizer for faster convergence.


# Task 3: Training Considerations
# It really depends on amound of data and the similarity of the tasks. If we have huge amount of data, 
# we can fune tune the whole model including the transformer layers. But if we have limited data,
# We freeze the lower layers of the model, including the Embedding layers and Lower transformer layers,
# because they are already well-trained on a large corpus data and shouldn't be updated during fine-tuning.
# However, we update the weights for upper transformer layers and the task-specific heads by fine-tuning on each task datasets.
# if data is very small and very limited, we freeze all layers of encoder, and only train the task-specific heads.

# Metrics for evaluation: For both classificaiton and sentiment analysis we can user Accuracy, Precision, Recall, and F1-score
# on the test data. For NER, we can use the F1-score, Precision, and Recall at the token level and for each class.

import torch
import torch.nn as nn
from transformers import AutoModel
from sentence_transformers import SentenceTransformer
from torch.optim import AdamW
import torch.nn.functional as F


class MultiTaskLearner(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", num_classes=2, sentiment_classes = 3, num_ner_labels=5):
        super().__init__()
        self.encoder = SentenceTransformer(model_name)
        vector_size = self.encoder.get_sentence_embedding_dimension() # Embedding Vector size
        
        # Task-Specific Heads: Fully connected NN
        self.classification_head = nn.Linear(vector_size, num_classes)  # Sentence Classification
        self.sentiment_head = nn.Linear(vector_size, sentiment_classes)  # Sentiment (Positive, Neutral, Negative)
        self.ner_head = nn.Linear(vector_size, num_ner_labels)  # NER 

    def forward(self, sentences, task="classification"):
        sentence_embeddings = self.encoder.encode(sentences, convert_to_tensor=True)  # Fixed-length embedding
        
        if task == "classification":
            return self.classification_head(sentence_embeddings)
        elif task == "sentiment":
            return self.sentiment_head(sentence_embeddings)
        elif task == "ner":
            return self.ner_head(sentence_embeddings)  # Token-level predictions
        else:
            raise ValueError("Task not supported!")




def train(model, training_data, learning_rate=2e-6, epochs=20):
    try:
        optimizer = AdamW(model.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            for data in training_data:
                sentences, labels, task_type = data  # Training data includes label and task type
                
                optimizer.zero_grad() # reset gradients
                outputs = model(sentences, task=task_type)
                
                if task_type == "classification" or task_type == "sentiment":
                    loss = F.cross_entropy(outputs, labels)  # outout 2D tensor [batch_size, num_classes]
                elif task_type == "ner": # output 3D tensor [batch_size, seq_len, num_labels]
                    loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), labels.view(-1))  # reshapes it into a 2D tensor
                else:
                    raise ValueError("Task not supported!")
                
                loss.backward()
                optimizer.step()

            print(f"Epoch number {epoch+1}: Loss = {loss.item():.3f}")

        print(f"Training completed after {epochs} epochs!")
    except Exception as e:
        print(f"Exception during training: {e}")

model = MultiTaskLearner()

sentences = ["Eagles won the game!", "bald eagle was seen in the sky today"]
classification_output = model(sentences, task="classification")
sentences = ["Eagles won the game, great job!", "Very disappointing game from Kensas city team."]
sentiment_output = model(sentences, task="sentiment")

print("Classification Output Shape:", classification_output.shape)  # [batch_size, num_classes]
print("Sentiment Output Shape:", sentiment_output.shape)  # [batch_size, 3]

# sample training data
training_data = [("Eagles won the game!", 1, "classification"), ("bald eagle was seen in the sky today", 0, "classification"),
                 ("Eagles won the game, great job!", 1, "sentiment"), ("Very disappointing game from Kensas city team.", 0, "sentiment"),
                 ("Eagles played in New Orleans today!", [1, 0, 0, 2, 0], "ner")]
train(model,training_data)