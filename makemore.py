import torch


START = "<S>"
END = "<E>"

names = open("names.txt").read().splitlines()
chars = sorted(set("".join(names)))
vocab = {strr: idx for idx, strr in enumerate(chars + [START, END])}
counts = torch.zeros((len(vocab), len(vocab)), dtype=torch.int32)

for name in names:
    name = [START] + list(name) + [END]
    
    for str1, str2 in zip(name, name[1:]):
        tok1, tok2 = vocab[str1], vocab[str2]
        counts[tok1, tok2] += 1

print(counts)