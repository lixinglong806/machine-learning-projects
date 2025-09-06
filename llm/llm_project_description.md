
## Project 1: Building LLM SMS Spam Classifier from scratch
Code in: [llm/llms-from-scratch/](./llm/llms-from-scratch/)

Run with:  
```bash
python llm/llms-from-scratch/funingtuning-for-text-classification.py
```

**Build LLMs model structure from scratch** and then **fine-tune** a pretrained **GPT-2** model with a classification head to detect **spam SMS messages** using the UCI SMS Spam Collection dataset.


## Highlights
- Dataset: UCI **SMS Spam Collection** (~5.5k messages)  
- Model: **GPT-2** frozen + trainable classification head  
- Tokenization: GPT-2 BPE (`tiktoken`)  
- Training: PyTorch `AdamW`, accuracy/loss tracking  
- Output: Classify text as **spam** or **ham**  


## Result
before finetuning:
<img width="221" height="54" alt="image" src="https://github.com/user-attachments/assets/76f9d8d8-cca0-4e5a-99ab-30a1eea1ceff" />
after finetuning:
<img width="254" height="61" alt="image" src="https://github.com/user-attachments/assets/b3526eb2-89ef-4b5d-bb23-86e7f141b2ef" />


## Project 2: Instruction-Tuned LLM for Text Generation
Code in: [llm/llms-from-scratch/](./llm/llms-from-scratch/)

Run with:  
```bash
python llm/llms-from-scratch/finetuning-to-follow-Instructions.py
```

This project explores **instruction tuning** for large language models (LLMs).  
We start with a pretrained GPT-2 model and fine-tune it on a dataset of human-written instructions (1.1k) and responses to align the model with task-following behavior.


## Dataset
- **Source**: [instruction-data.json](https://github.com/lixinglong806/machine-learning-projects/blob/main/llm/llms-from-scratch/instruction-data.json)  
- **Size**: ~1,000+ instruction–response pairs  
- **Structure**:
  - `instruction`: task description  
  - `input`: optional context  
  - `output`: ground-truth response  

Example:
```json
{
  "instruction": "Translate 'good morning' to French.",
  "input": "",
  "output": "Bonjour"
}
```

## Model
- **Base model**: GPT-2 (124M / 355M / 774M / 1558M)  
- **Tokenizer**: GPT-2 BPE (`tiktoken`)  
- **Objective**: Fine-tune GPT-2 with causal LM loss, supervised by *(instruction + input → output)* pairs  
- **Training setup**:  
  - Optimizer: AdamW  
  - Context length: 1024  
  - Epochs: 2–3  
  - Batch size: 8  

## Tech Stack
- **Framework**: PyTorch  
- **Data handling**: `torch.utils.data.Dataset`, `DataLoader`  
- **Tokenizer**: `tiktoken`  
- **Pretrained weights**: HuggingFace GPT-2 (via helper `download_and_load_gpt2`)

## Evaluation (LLM-as-a-Judge via Ollama)

- **Judge model:** `llama3.2` (running locally with **Ollama**)
- **Protocol:** rubric-based scoring of model responses on the **test set** (deterministic settings: `seed=123`, `temperature=0`, `num_ctx=2048`)
- **Result: <img width="230" height="38" alt="image" src="https://github.com/user-attachments/assets/54595b65-fd47-4793-ad47-b7794562ba0f" />



