from pathlib import Path

import torch


SEP = "."


def read_inputs(dataset_path: Path) -> list[str]:
    return open(dataset_path).read().splitlines()


def build_vocab(inputs: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    chars = sorted(set("".join(inputs)))
    stoi = {s: i for i, s in enumerate([SEP] + chars)}
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


def build_bigram_model(inputs: list[str], stoi: dict[str, int]) -> torch.Tensor:
    counts = torch.zeros((len(stoi), len(stoi)), dtype=torch.int32)

    for inpt in inputs:
        inpt = [SEP] + list(inpt) + [SEP]

        for str1, str2 in zip(inpt, inpt[1:]):
            idx1, idx2 = stoi[str1], stoi[str2]
            counts[idx1, idx2] += 1

    return counts


if __name__ == "__main__":
    inputs = read_inputs(Path("names.txt"))
    stoi, itos = build_vocab(inputs)
    counts = build_bigram_model(inputs, stoi)
    print(counts)
