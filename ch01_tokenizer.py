# %%

import os
import urllib.request

if not os.path.exists("the-verdict.txt"):
    url = ("https://raw.githubusercontent.com/rasbt/"
           "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
           "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)

# %%
import re

class SimpleTokenV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    

    def decode(self, ids):
        text = " ".join([
            self.int_to_str[i]
            for i in ids
        ])
        text = re.sub(f'\s+([,.?!"()\'])', r'\1', text)
        return text

# %%
text = "Hello, world. Is this -- a text?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)

# %%

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
all_words = sorted(set([item.strip() for item in preprocessed if item.strip()]))
vocab = {token:integer for integer, token in enumerate(all_words)}
print(len(preprocessed))
# %%
tokenizer = SimpleTokenV1(vocab)
text =  """"It's the last he painted, you know," 
           Mrs. Gisburn said with pardonable pride."""
ids = tokenizer.encode(text)
print(ids)
# %%
tokenizer.decode(ids)
# %%

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(['<|endoftext|>', '<|unk|>'])

vocab = {token:integer for integer,token in enumerate(all_tokens)}

# %%

for i, item in enumerate(list(vocab.items())[-5:]):
    print(item)
# %%

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(f'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [
            item if item in self.str_to_int else "<|unk|>" 
            for item in preprocessed
        ]
        
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(f'\s+([,.:;?!"()\'])', r'\1', text)
        return text
# %%
tokenizer = SimpleTokenizerV2(vocab)
text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palace."

text = "<|endoftext|>".join((text1, text2))

print(text)
# %%
tokenizer.encode(text)
# %%
tokenizer.decode(tokenizer.encode(text))

# %%
tokenizer.str_to_int

# %%
