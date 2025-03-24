# Character-level-Language-Model
Generating Shakespeare Using a Character-level Language Model

![Shakespeare](https://i.imgur.com/81YZuel.jpg)

## Overview
This project implements a character-level language model to generate Shakespearean-style text using PyTorch. The model is trained on a dataset of Shakespeare's works and utilizes a recurrent neural network (GRU) to predict the next character in a sequence.

## Features
- Character-based language modeling
- Implemented using PyTorch
- GRU-based recurrent neural network
- Generates text in Shakespearean style
- Supports GPU acceleration

## Installation
To run this project, install the required dependencies:

```bash
pip install torch unidecode
```

## Dataset
The dataset consists of Shakespeare's texts obtained from:

```
https://github.com/tau-nlp-course/NLP_HW2/raw/main/data/shakespeare.txt
```

We preprocess the data using the `unidecode` package to convert Unicode characters to ASCII.

## Model Architecture
The model consists of:
1. **Embedding Layer**: Maps characters to learned vector representations.
2. **GRU Layer**: Processes sequences and captures contextual dependencies.
3. **Output Layer**: Predicts the probability distribution of the next character.

## Training
The training process involves:
1. Converting text into numerical tensors.
2. Using a loss function to minimize prediction errors.
3. Updating the model weights using backpropagation.

Run the training loop using:

```python
for epoch in range(num_epochs):
    inp, target = random_training_set()
    loss = train(inp, target)
    print(f"Epoch {epoch}, Loss: {loss}")
```

## Text Generation
Once trained, the model can generate Shakespeare-like text by predicting characters iteratively:

```python
print(evaluate(prime_str="To be, or not to be", predict_len=200))
```

## Advantages
### Character-Based Language Model
* **Handles Out-of-Vocabulary (OOV) Words:** Character-based models work at the level of
individual characters, so they can handle words that the model has never seen before, such as slang,
typos, or newly created words. This makes them particularly robust for dynamic and noisy text
data, like social media or usernames.
* **Effective for Morphologically Rich Languages:** In languages with complex word structures
(e.g., Turkish or Finnish), character-based models can naturally learn the relationships between
prefixes, suffixes, and root words, making them highly effective.
* **Compact Vocabulary:** Since the model uses individual characters instead of words, the vocabulary size is much smaller. This reduces memory requirements and simplifies the model.
* **Flexible with Input Lengths:** Character-based models can handle unusually long or short words,
such as scientific terms, URLs, or rare names, without issues.

### Word-Based Language Model
* **Faster Training and Inference:** Word-based models process entire words, which reduces the
number of steps required to analyze text compared to character-level models. This makes them
faster and more computationally efficient.
* **Better Semantic Understanding:** Word embeddings used in word-based models often encode
rich semantic and syntactic information, allowing these models to capture the meaning and relationships between words more effectively.
* **Efficient Sequence Lengths:** A word-based model requires fewer tokens to represent a sentence
compared to a character-based model. This makes processing long texts more efficient.
* **High Performance on Standard Text:** For tasks involving formal or standard text, such as
news articles or essays, word-based models often perform better because they can directly leverage
the meaning of entire words

## Usage
To use the model, ensure all dependencies are installed and run the Jupyter Notebook with:

```
Runtime → Run All
```

For GPU acceleration:
```
Runtime → Change runtime type → Select GPU
```

## License
This project is for educational purposes. Feel free to modify and extend it!

---

🚀 Happy Coding!

