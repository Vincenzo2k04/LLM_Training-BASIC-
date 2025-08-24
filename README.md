# Scratch-Built Language Model (SLM) based on GPT-2

This repository contains the complete code for building, training, and fine-tuning a language model from the ground up. This project serves as an in-depth, educational exploration into the architecture and mechanics of transformer-based language models. While it exhibits characteristics of a Large Language Model (LLM), given its training on a small dataset and its utilization of GPT-2 (355M) weights, it is more accurately described as a Small Language Model (SLM).

The entire implementation is meticulously guided by the principles and code outlined in **Sebastian Raschka's** insightful book, *"Build a Large Language Model From Scratch."*

---

## 📜 Project Overview

This project traces the complete lifecycle of a language model, from initial data processing to task-specific fine-tuning. It is designed to be a transparent and accessible resource for understanding the core components that power modern NLP models.

The model is built using PyTorch and leverages the `tiktoken` library for efficient tokenization, mirroring the processes used in state-of-the-art models.

---

## ⚙️ Code Pathway and Workflow

The codebase is structured to follow a logical progression, building complexity at each step:

1.  **Data Ingestion & Preprocessing**: The model's journey begins with fetching a small text dataset ("the-verdict.txt").
2.  **Tokenization**: Initially, a simple regex-based tokenizer is built to understand the fundamentals. The project then transitions to OpenAI's `tiktoken` library with the `gpt2` encoding for more robust and efficient text-to-token conversion.
3.  **Data Loading**: The tokenized data is structured into input-target pairs using a sliding window approach and loaded efficiently using PyTorch's `DataLoader`.
4.  **Core Architecture Implementation**:
    * **Embeddings**: Both token and positional embeddings are created to capture semantic meaning and word order.
    * **Self-Attention**: The core of the transformer, the self-attention mechanism, is built from scratch. This is then evolved into a more sophisticated **Causal Multi-Head Attention** mechanism, which allows the model to weigh the importance of different tokens in the input sequence without looking ahead.
    * **Transformer Block**: The attention mechanism is integrated with feed-forward neural networks, layer normalization, and residual connections to form a complete `TransformerBlock`.
5.  **GPT Model Assembly**: Multiple `TransformerBlock` layers are stacked to create the final `GPTModel` architecture, complete with an output head to predict the next token.
6.  **Pre-training**: The model is pre-trained on the initial text dataset. This phase involves training the model to predict the next word in a sequence, allowing it to learn grammar, context, and basic language patterns.
7.  **Fine-Tuning for Classification**: After pre-training, the model is adapted for a specific downstream task: **spam classification**. The model's head is replaced with a classification layer, and it is fine-tuned on a labeled spam/ham dataset.
8.  **Instruction Fine-Tuning**: To enhance the model's ability to follow commands, it undergoes instruction fine-tuning on a curated dataset of instruction-response pairs.

---

## 🚀 How It Works: A Quick Sample

Here’s a brief example of how to use the fine-tuned model to classify a piece of text after loading the appropriate weights.

```python
# main.py

import torch
import tiktoken
from your_model_file import GPTModel # Assuming your model class is saved here

# --- Configuration and Model Loading ---
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = tiktoken.get_encoding("gpt2")

# Load your fine-tuned model configuration
# Ensure this matches the configuration used for saving the model
BASE_CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,
    "qkv_bias": True
}

# Initialize the model and load the fine-tuned weights
model = GPTModel(BASE_CONFIG)
model.load_state_dict(torch.load("text_classifier.pth", map_location=device))
model.to(device)
model.eval()

# --- Classification Function ---
def classify_text(text, model, tokenizer, device, max_length=120, pad_token_id=50256):
    """
    Classifies a given text as 'spam' or 'not spam'.
    """
    inputs_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[0]

    # Truncate and pad the input
    inputs_ids = inputs_ids[:min(max_length, supported_context_length)]
    inputs_ids += [pad_token_id] * (max_length - len(inputs_ids))
    inputs_tensor = torch.tensor(inputs_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(inputs_tensor)[:, -1, :] # Use the last token's output for classification
        predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"

# --- Example Usage ---
text_to_classify = "You are a winner you have been specially selected to receive $1000 cash or a $2000 award"
prediction = classify_text(text_to_classify, model, tokenizer, device)
print(f"The text is classified as: {prediction}")
```

---

## 📋 Requirements

To run this project, you will need the following libraries:

* `torch`
* `tiktoken`
* `pandas`
* `numpy`
* `matplotlib`
* `tensorflow` (for loading OpenAI's pretrained weights)
* `tqdm`

You can install them using pip:
`pip install torch tiktoken pandas numpy matplotlib tensorflow tqdm`

---

## ⚠️ Shortcomings & Limitations

This model is an educational tool and has several key limitations:

* **Small Dataset**: It was trained on a very small corpus ("the-verdict.txt"), which severely limits its knowledge base and ability to generate diverse and contextually rich text.
* **Limited Scale**: The model is based on the GPT-2 Medium (355M) architecture. Due to resource constraints, it cannot scale to the size of modern LLMs, which have billions or even trillions of parameters.
* **Basic Reasoning**: The model's ability to perform complex reasoning, inference, or creative tasks is minimal due to the limitations mentioned above.
* **Overfitting**: As seen in the training graphs, the model begins to overfit on the small training dataset, with the validation loss plateauing or increasing while the training loss continues to decrease.

---

## 🙏 Credits and Acknowledgements

This entire project was made possible by the comprehensive and brilliantly structured guide, **"Build a Large Language Model From Scratch"** by **Sebastian Raschka**. His work provided the foundational knowledge, code, and methodology that this repository is built upon. I extend my sincerest gratitude for his invaluable contribution to the machine learning community.
