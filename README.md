# Transformer-based NLP Project

This project uses pre-trained transformer models (such as DistilBERT and GPT-2) to perform masked language modeling (MLM) and text generation tasks. The model is fine-tuned on a custom dataset to predict missing words in a sentence and generate new text based on a prompt.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Model Training](#model-training)
4. [Usage](#usage)

## Project Overview

This project implements two key functionalities:
1. **Masked Language Modeling (MLM)** using DistilBERT for pretraining on a custom text dataset.
2. **Text Generation** using GPT-2, which can generate creative text based on a provided prompt.

### Technologies Used:
- Python 3
- PyTorch
- Hugging Face Transformers
- pandas
- scikit-learn

## Installation

Follow these steps to set up the project locally.

### 1. Clone the Repository:
```bash
[git clone https://github.com/Saketh-Vadlamudi/LanguageModelling.git]
cd LanguageModelling
```
### 2. Install Dependencies:
```bash
pip install -r requirements.txt
```


## Model Training:
```
To train the model on your dataset:
python train_model.py
```

## Usage
```
To generate text using the pre-trained GPT-2 model:
python generate_text.py

Example usage:
python generate_text.py --text "NLP is based on transformers."
```
