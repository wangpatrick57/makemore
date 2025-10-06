import torch


SEP = "."

names = open("names.txt").read().splitlines()
chars = sorted(set("".join(names)))
stoi = {s: i for i, s in enumerate([SEP] + chars)}
itos = {i: s for s, i in stoi}
counts = torch.zeros((len(stoi), len(stoi)), dtype=torch.int32)

for name in names:
    name = [SEP] + list(name) + [SEP]

    for str1, str2 in zip(name, name[1:]):
        idx1, idx2 = stoi[str1], stoi[str2]
        counts[idx1, idx2] += 1

print(counts)
