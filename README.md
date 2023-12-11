# Improving Question Generation with Fine-tuning of MT5

## Result Link
The fine-tuned Model is uploaded to hugging face at https://huggingface.co/ZihanXie/QuestionGeneration

## How to use it
### Load model directly
Use transformer package in python and import below packages
```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("ZihanXie/QuestionGeneration")
model = AutoModelForSeq2SeqLM.from_pretrained("ZihanXie/QuestionGeneration")
```

### Introduction
My project that uses a fine-tuned T5 model aims to address the problem of automating the process of generating questions from a pair of context and answer. Traditionally, generating questions from text has been a manual task performed by humans, and is often a time-consuming and labor-intensive process. We use NLP techniques to automate the task of generating questions from text at scale, and to ensure that the questions generated are accurate, relevant, and appropriate for the intended use case.

### Problem
Question generation is a natural language processing task that involves generating a question from a given context and answer. The goal of this task is to generate a question that is relevant to the context and answer, and that elicits information that is not explicitly stated in the context. This task is particularly challenging because it requires the model to understand the relationship between the context and answer and to generate a question that is both grammatically correct and semantically meaningful. Furthermore, the generated question should not simply repeat the context or answer but should instead probe deeper into the underlying meaning and implications of the text.

### Overview
<img width="1000" alt="1" src="https://github.com/ZihanXie/CSCI596_Final_Project/assets/112039431/1b676dfe-6069-4fe0-94c4-854b8d34c8a8">

### T5 Model
![2](https://github.com/ZihanXie/CSCI596_Final_Project/assets/112039431/f1a09924-9fd4-4d90-ac1c-a25e3321ba3d)
I employed the MT5 model, a multilingual variant of T5. My choice for MT5 is motivated by its ability to process training data in various languages, which would be particularly advantageous if additional GPU resources were available.

### Dataset
#### SQuAD
The Stanford Question Answering Dataset (SQuAD) is a popular benchmark dataset for machine reading comprehension tasks. It con- sists of a large collection of Wikipedia articles and a set of question-answer pairs, where each question corresponds to a span of text in the article. The goal is for a machine learning model to read the article and provide the cor- rect answer to each question.
#### DRCD
The Delta Reading Comprehension Dataset (DRCD) is a Chinese language benchmark dataset for machine reading comprehension tasks. It consists of a large collection of news articles and web pages, with over 10,000 man- ually annotated questions and corresponding answers. DRCD was created to evaluate the ability of machine learning models to under- stand and reason over Chinese text, which is a significant challenge due to the complexity of the language and the lack of large-scale annotated datasets.

### Expected Result
<img width="600" alt="Screenshot 2023-11-25 at 9 21 55 PM" src="https://github.com/ZihanXie/CSCI596_Final_Project/assets/112039431/26c50e28-4e68-4b34-9f85-cf52970376ff">

