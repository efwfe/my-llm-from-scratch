# %% 
# 1. llm ft approches
# 2. preparate dataset for text classification
# 3. modify pretrained llm for ft
#  4. evaluate the accuracy of llm classifier
#  5. a fine-tuned llm to classify new data 
import torch
import urllib.request
import zipfile
import os
from pathlib import Path

url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
zip_path = "sms_spam_collection.zip"
extracted_path = "sms_spam_collection"
data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"


# %%
def download_and_unzip_spam_data(
    url,
    zip_path,
    extracted_path,
    data_file_path
):
    if data_file_path.exists():
        print(f"{data_file_path} already exists. Skipping download and extraction.")
        return 

    with urllib.request.urlopen(url) as response:
        with open(zip_path, 'wb') as out_file:
            out_file.write(response.read())
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extracted_path)
    
    originmal_file_path = Path(extracted_path) / "SMSSpamCollection"
    os.rename(originmal_file_path, data_file_path)
    print(f"File downloaded and saved as {data_file_path}")

# %%
download_and_unzip_spam_data(
    url,
    zip_path,
    extracted_path,
    data_file_path
)
# %%

import pandas as pd

df = pd.read_csv(
    data_file_path,
    sep="\t",
    header=None,
    names=["Label", "Text"] 
    
)

df.head()
# %%
df['Label'].value_counts()
# %%
def create_balanced_dataset(df):
    num_spam = df[df['Label'] == 'spam'].shape[0]
    ham_subset = df[df['Label'] == 'ham'].sample(
        num_spam, random_state=123
    )
    balanced_df = pd.concat([
        ham_subset,
        df[df['Label'] == 'spam']
    ])
    return balanced_df

balanced_df = create_balanced_dataset(df)
print(balanced_df['Label'].value_counts())
# %%
balanced_df['Label'] = balanced_df['Label'].map({
    "ham": 0,
    "spam": 1
})


# %%

def random_split(df, train_frac, validation_frac):
    
    df = df.sample(
        frac=1,
        random_state=123
    ).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)
    
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    return train_df, validation_df, test_df


# %%
train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
# %%
train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)
# %% Create data loaders

import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

print(tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}))
# %%
import torch
from torch.utils.data import Dataset

 
class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [
             tokenizer.encode(text) for text in self.data['Text']
        ]
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]
        
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]
    
    
    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]['Label']
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )
        
        
    def __len__(self):
        return len(self.data)
    
    
    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length
# %%
train_dataset = SpamDataset(
    csv_file="train.csv",
    max_length=None,
    tokenizer=tokenizer
)
# %%
train_dataset.max_length
# %%
val_dataset = SpamDataset(
    "validation.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)

test_dataset = SpamDataset(
    "test.csv",
    max_length=train_dataset.max_length,
    tokenizer=tokenizer
)
# %%
from torch.utils.data import DataLoader

num_workers = 0
batch_size = 8
torch.manual_seed(123)


train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size= batch_size,
    num_workers=num_workers,
    drop_last=False
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False
)
# %%
print(f"Train batches: {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches {len(test_loader)}")
# %% Initializaing a model with pretrained weights
CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"
BASE_CONFIG = {
    "vocab_size": 50257,          #1
    "context_length": 1024,       #2
    "drop_rate": 0.0,             #3
    "qkv_bias": True              #4
}
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])
# %%
from gpt_download import download_and_load_gpt2
from ch05_pretraining import GPTModel, load_weiths_into_gpt

model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(
    model_size=model_size, models_dir="gpt2"
)

model  = GPTModel(BASE_CONFIG)
load_weiths_into_gpt(model, params)
model.eval()


# %%
from ch04_impl_gpt import generate_text_simple
from ch05_pretraining import text_to_token_ids, token_ids_to_text

text_1 = "Every effort moves you"
token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_1, tokenizer),
    max_new_tokens=15,
    context_size=BASE_CONFIG['context_length']
)

print(token_ids_to_text(token_ids, tokenizer))
# %%
text_2 = (
    "Is the following text 'spam'? Answer with 'yes' or 'no':"
    "'You are winner you have been specially selected to receive $1000 cash or a $2000 award."
)


token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(text_2, tokenizer=tokenizer),
    max_new_tokens=23,
    context_size=BASE_CONFIG['context_length']
)

print(token_ids_to_text(token_ids, tokenizer))

# %% adding a classification head
for param in model.parameters():
    param.requires_grad = False
    
    
torch.manual_seed(123)
num_classes = 2

model.out_head = torch.nn.Linear(
    in_features=BASE_CONFIG['emb_dim'],
    out_features=num_classes
)
# %%
# fine-tuning additional layers can noticeably improve the predictive performace of the model

for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True

for param in model.final_norm.parameters():
    param.requires_grad = True
    
    

# %%
inputs = tokenizer.encode("Do you have time")
inputs = torch.tensor(inputs).unsqueeze(0)
print(inputs.shape)
# %%
with torch.no_grad():
    outputs = model(inputs)

print(outputs)
print(outputs.shape)
# %%
# extract last output token
print(f"Last output token: {outputs[:, -1, :]}")
# %%
# due to the causal attention mask, preventing tokens from attending to future tokens.
# the values in the cells represent attention scores; the last token is the only one 
# that computes attention scores from all preceding tokens.

# %% Calculation the classification loss and accuracy
# 
probas = torch.softmax(outputs[:, -1, :], dim=-1)
label = torch.argmax(probas)
print(label.item())
# %%
logits = outputs[:, -1, :]
print(torch.argmax(logits).item())
# %%

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    
    currect_predictions, num_examples = 0, 0
    
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            with torch.no_grad():
                logits = model(input_batch)[:, -1, :]
            
            predicted_labels = torch.argmax(logits, dim=-1)
            num_examples += predicted_labels.shape[0]
            currect_predictions += (
                (predicted_labels == target_batch).sum().item()
            )
        else:
            break
    return currect_predictions / num_examples
# %%
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model.to(device)

torch.manual_seed(123)

train_accuracy = calc_accuracy_loader(
    train_loader, model, device, num_batches=10
)

val_accuracy = calc_accuracy_loader(
    val_loader, model, device, num_batches=10
)

test_accuracy = calc_accuracy_loader(
    test_loader, model, device, num_batches=10
)


print(f"Train accuracy: {train_accuracy}")
print(f"Validation accuracy : {val_accuracy}")
print(f"Test accuracy: {test_accuracy}")
# %%


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)[:, -1, :]
    loss =torch.nn.functional.cross_entropy(logits, target_batch)
    return loss
# %%
def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    
    if len(data_loader) == 0:
        return float('nan')
    
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break
    
    return total_loss / num_batches
# %%

with torch.no_grad():
    train_loss = calc_accuracy_loader(train_loader, model, device, num_batches=5)
    val_loss =calc_accuracy_loader(val_loader, model, device, num_batches=5)
    tes_loss = calc_accuracy_loader(test_loader, model, device, num_batches=5)
    

print(f"Training loss: {train_loss:.3f}")
print(f"Valication loss: {val_loss:.3f}")
print(f"Test loss: {tes_loss:.3f}")

# %% Fine-tuning the model on supervised data

def train_classifier_simple(
    model, 
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs,
    eval_freq,
    eval_iter
):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1
    
    for epoch in range(num_epochs):
        model.train()
        
        for input_batch, target_batch in train_loader:
            # reset loss gradients from previous batch
            optimizer.zero_grad()
            # calculate loss 
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            # backward 
            loss.backward()
            optimizer.step()
            
            examples_seen += input_batch.shape[0]
            global_step += 1
            
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model,train_loader, val_loader,
                    device, eval_iter
                )
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch + 1} . Step {global_step: 06d} : Train loss {train_loss:.3f}, Val loss: {val_loss:.3f}")
                
        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter
        )
        train_accs.append(train_accuracy)
        val_accs.append(val_accs)
        print(f"Training accuracy: {train_accuracy:.2f}")
        print(f"Validation accuracy: {val_accuracy:.2f}")
    
    return train_losses, val_losses, train_accs, val_accs, examples_seen


            
# %%
def evaluate_model(model, train_loader, val_loader,  device, eval_iter):
    model.eval()
    
    with torch.no_grad():
        train_loss =calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader,
            model,
            device,
            eval_iter
        )
        model.train()
    return train_loss, val_loss
# %%
import time
start_time = time.time()

torch.manual_seed(123)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)
num_epcohs = 5

train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    num_epochs=num_epcohs,
    eval_freq=50,
    eval_iter=5
)

end_time = time.time()
print(f"Training completed in {(end_time - start_time) / 60 : .2f}")
# %%
import  matplotlib.pyplot as plt

def plot_values(
    epochs_seen,
    example_seen,
    train_values,
    val_values,
    label="loss"
):
    fig, ax1 = plt.subplots(figsize=(5,3))
    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(epochs_seen, val_values, linestyle='-.', label=f"Validation {label}")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()
    
    ax2 = ax1.twiny()
    ax2.plot(example_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")
    
    fig.tight_layout()
    plt.savefig(f"{label}-plot.gif")
    plt.show()

epochs_tensor = torch.linspace(0, num_epcohs, len(train_losses))
example_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

plot_values(epochs_tensor, example_seen_tensor, train_losses, val_losses)
    
# %%
epochs_tensor = torch.linspace(0, num_classes, len(train_accs))
example_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))

plot_values(
    epochs_tensor,
    example_seen_tensor,
    train_accs,
    val_accs,
    label="accuracy"
)
# %%
train_accs
# %%
test_accuracy = calc_accuracy_loader(test_loader, model, device)
print(f"Test accuracy: {test_accuracy:.4f}")
# %% Using the LLM as a spam classifier

def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()
    
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[1]
    
    input_ids = input_ids[:min(max_length, supported_context_length)]
    
    input_ids += [pad_token_id * (max_length - len(input_ids))]
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    
    predicted_label = torch.argmax(logits, dim=-1).item()
    return 'spam' if predicted_label == 1 else "not spam"

# %%
text_1 = (
    "You are a winner you have been specially selected to receive $1000 cash or a $2000 award."
)

print(classify_review(
    text_1,
    model,tokenizer, device, max_length=train_dataset.max_length
))
# %%
torch.save(model.state_dict(), "review_classifier.pth")

# %%
model_state_dict = torch.load("review-classifier.pth", map_location=device)
model.load_state_dict(model_state_dict)