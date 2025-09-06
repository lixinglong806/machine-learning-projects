import json
import os
import urllib

import torch
from torch.utils.data import Dataset
import tiktoken
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import psutil
import urllib.request
from gpt_download import download_and_load_gpt2
from llm_models import *
from load_pretrained_weights import *
from llm_pretrain import *

# load finetune datasets
def download_and_load_file(file_path, url):

    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

    # The book originally contained this unnecessary "else" clause:
    #else:
    #    with open(file_path, "r", encoding="utf-8") as file:
    #        text_data = file.read()

    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    return data


file_path = "instruction-data.json"
url = (
    "https://github.com/lixinglong806/machine-learning-projects"
    "/blob/main/llm/llms-from-scratch/instruction-data.json"
)

data = download_and_load_file(file_path, url)
print("Number of entries:", len(data))
print("Example entry:\n", data[50])
print("Another example entry:\n", data[999])

# split train/test/val dataset
train_portion = int(len(data) * 0.85)  # 85% for training
test_portion = int(len(data) * 0.1)    # 10% for testing
val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

train_data = data[:train_portion]
test_data = data[train_portion:train_portion + test_portion]
val_data = data[train_portion + test_portion:]

print("Training set length:", len(train_data))
print("Validation set length:", len(val_data))
print("Test set length:", len(test_data))


# organize data into training batches
## original dataset -> formatted data
def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            instruction_plus_input = format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


tokenizer = tiktoken.get_encoding("gpt2")
print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))


def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # Find the longest sequence in the batch
    batch_max_length = max(len(item)+1 for item in batch)

    # Pad and prepare inputs and targets
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # Add an <|endoftext|> token
        new_item += [pad_token_id]
        # Pad sequences to max_length
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])  # Truncate the last token for inputs
        targets = torch.tensor(padded[1:])  # Shift +1 to the right for targets

        # New: Replace all but the first padding tokens in targets by ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # New: Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # Convert list of inputs and targets to tensors and transfer to target device
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


# Creating data loaders for an instruction dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

customized_collate_fn = partial(
    custom_collate_fn,
    device=device,
    allowed_max_length=1024
)

# Creating data loaders for an instruction dataset
num_workers = 0
batch_size = 8
torch.manual_seed(123)

train_dataset = InstructionDataset(train_data, tokenizer)
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=True,
    drop_last=True,
    num_workers=num_workers
)
val_dataset = InstructionDataset(val_data, tokenizer)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

test_dataset = InstructionDataset(test_data, tokenizer)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    collate_fn=customized_collate_fn,
    shuffle=False,
    drop_last=False,
    num_workers=num_workers
)

## see the dims of input and target batches
# print("Train loader:")
# for inputs, targets in train_loader:
#     print(inputs.shape, targets.shape)
for inputs, targets in train_loader:
    print('some input = ', inputs[0])
    print('corresponding target = ', targets[0])
    break

# load a pretrained LLM
BASE_CONFIG = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "drop_rate": 0.0,        # Dropout rate
    "qkv_bias": True         # Query-key-value bias
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

CHOOSE_MODEL = "gpt2-medium (355M)"

BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(
    model_size=model_size,
    models_dir="gpt2"
)

model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)
model.eval();

# check if the model can generate good answer to our input
torch.manual_seed(123)
input_text = format_input(val_data[0])
print('input = ')
print(input_text)

token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256,
)
generated_text = token_ids_to_text(token_ids, tokenizer)
print('answer from model to input = ')
print(generated_text)

# finetuning LLM on instruction data
model.to(device)

torch.manual_seed(123)

# def calc_loss_batch(input_batch, target_batch, model, device):
#     input_batch, target_batch = input_batch.to(device), target_batch.to(device)
#     logits = model(input_batch)[:, -1, :]  # Logits of last output token
#     loss = torch.nn.functional.cross_entropy(logits, target_batch)
#     return loss
#
# def calc_loss_loader(data_loader, model, device, num_batches=None):
#     total_loss = 0.
#     if len(data_loader) == 0:
#         return float("nan")
#     elif num_batches is None:
#         num_batches = len(data_loader)
#     else:
#         # Reduce the number of batches to match the total number of batches in the data loader
#         # if num_batches exceeds the number of batches in the data loader
#         num_batches = min(num_batches, len(data_loader))
#     for i, (input_batch, target_batch) in enumerate(data_loader):
#         if i < num_batches:
#             loss = calc_loss_batch(input_batch, target_batch, model, device)
#             total_loss += loss.item()
#         else:
#             break
#     return total_loss / num_batches
#
# with torch.no_grad():
#     train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
#     val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)
#
# print("Training loss:", train_loss)
# print("Validation loss:", val_loss)

# import time
#
# start_time = time.time()
#
# torch.manual_seed(123)
#
# optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)
#
# num_epochs = 2
#
# train_losses, val_losses, tokens_seen = train_model_simple(
#     model, train_loader, val_loader, optimizer, device,
#     num_epochs=num_epochs, eval_freq=5, eval_iter=5,
#     start_context=format_input(val_data[0]), tokenizer=tokenizer
# )
#
# end_time = time.time()
# execution_time_minutes = (end_time - start_time) / 60
# print(f"Training completed in {execution_time_minutes:.2f} minutes.")

# save (and load) model
# import re
# file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth"
# torch.save(model.state_dict(), file_name)
# print(f"Model saved as {file_name}")

# Load model via
model.load_state_dict(torch.load("gpt2-medium355M-sft.pth"))


# # plot train loss vs epoch
# def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
#     fig, ax1 = plt.subplots(figsize=(5, 3))
#
#     # Plot training and validation loss against epochs
#     ax1.plot(epochs_seen, train_losses, label="Training loss")
#     ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
#     ax1.set_xlabel("Epochs")
#     ax1.set_ylabel("Loss")
#     ax1.legend(loc="upper right")
#     ax1.xaxis.set_major_locator(MaxNLocator(integer=True))  # only show integer labels on x-axis
#
#     # Create a second x-axis for tokens seen
#     ax2 = ax1.twiny()  # Create a second x-axis that shares the same y-axis
#     ax2.plot(tokens_seen, train_losses, alpha=0)  # Invisible plot for aligning ticks
#     ax2.set_xlabel("Tokens seen")
#
#     fig.tight_layout()  # Adjust layout to make room
#     plt.savefig("loss-plot.pdf")
#     plt.show()
#
# epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
# plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


print('input = ')
print(input_text)

token_ids = generate(
    model=model,
    idx=text_to_token_ids(input_text, tokenizer),
    max_new_tokens=35,
    context_size=BASE_CONFIG["context_length"],
    eos_id=50256,
)
generated_text = token_ids_to_text(token_ids, tokenizer)
print('after finetuning, answer from model to input = ')
print(generated_text)

# evaluate model
## generate answer for dataset and save them
for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
    input_text = format_input(entry)
    token_ids = generate(
        model=model,
        idx=text_to_token_ids(input_text, tokenizer).to(device),
        max_new_tokens=256,
        context_size=BASE_CONFIG["context_length"],
        eos_id=50256
    )
    generated_text = token_ids_to_text(token_ids, tokenizer)
    response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

    test_data[i]["model_response"] = response_text


with open("instruction-data-with-response.json", "w") as file:
    json.dump(test_data, file, indent=4)  # "indent" for pretty-printing

# use ollma to run other model like famous opensource models
def check_if_running(process_name):
    running = False
    for proc in psutil.process_iter(["name"]):
        if process_name in proc.info["name"]:
            running = True
            break
    return running

ollama_running = check_if_running("ollama")

# optional , load test data we saved before
# if not ollama_running:
#     raise RuntimeError("Ollama not running. Launch ollama before proceeding.")
# print("Ollama running:", check_if_running("ollama"))
#
# file_path = "instruction-data-with-response.json"
#
# with open(file_path, "r") as file:
#     test_data = json.load(file)



def query_model(
    prompt,
    model="llama3",
    url="http://localhost:11434/api/chat"
):
    # Create the data payload as a dictionary
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "options": {     # Settings below are required for deterministic responses
            "seed": 123,
            "temperature": 0,
            "num_ctx": 2048
        }
    }


    # Convert the dictionary to a JSON formatted string and encode it to bytes
    payload = json.dumps(data).encode("utf-8")

    # Create a request object, setting the method to POST and adding necessary headers
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST"
    )
    request.add_header("Content-Type", "application/json")

    # Send the request and capture the response
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # Read and decode the response
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data

def generate_model_scores(json_data, json_key, model="llama3"):
    scores = []
    for entry in tqdm(json_data, desc="Scoring entries"):
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry[json_key]}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
            f"NOTICE: ONLY Respond with the integer number."
        )
        score = query_model(prompt, model)
        try:
            scores.append(int(score))
        except ValueError:
            print(f"Could not convert score: {score}")
            continue

    return scores


scores = generate_model_scores(test_data, "model_response", model="llama3.2")
print(f"Number of scores: {len(scores)} of {len(test_data)}")
print(f"Average score: {sum(scores)/len(scores):.2f}\n")