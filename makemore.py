import torch


START = "<S>"
END = "<E>"

names = open("names.txt").read().splitlines()
chars = sorted(set("".join(names)))
stoi = {s: i for i, s in enumerate(chars + [START, END])}
counts = torch.zeros((len(stoi), len(stoi)), dtype=torch.int32)

for name in names:
    name = [START] + list(name) + [END]

    for str1, str2 in zip(name, name[1:]):
        idx1, idx2 = stoi[str1], stoi[str2]
        counts[idx1, idx2] += 1

print(counts)
