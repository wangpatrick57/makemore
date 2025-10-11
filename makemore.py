from pathlib import Path

import torch


SEP = "."


def read_inputs(dataset_path: Path) -> list[str]:
    """
    The path should point to a file where each line contains an example input of what
    we want to generate.
    """
    return open(dataset_path).read().splitlines()


def build_vocab(inputs: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """
    Returns a mapping from each unique character that appears in any input to a
    canonical integer representing that character. Includes the separator
    character (SEP) as well, representing the start/end of an input. Also returns
    the reverse mapping.

    As a convention, we sort the characters alphabetically and put SEP last.
    """
    chars = sorted(set("".join(inputs)))
    stoi = {s: i for i, s in enumerate([SEP] + chars)}
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


def build_bigram_model(inputs: list[str], stoi: dict[str, int]) -> torch.Tensor:
    """
    Returns the bigram model, which is an NxN Tensor where N is the vocab size. Each
    row (i, j) of the Tensor is a probability distribution representing the probability
    of j appearing after i amongst the inputs.
    """
    bigram = torch.zeros((len(stoi), len(stoi)), dtype=torch.int32)

    for inpt in inputs:
        inpt = [SEP] + list(inpt) + [SEP]

        for str1, str2 in zip(inpt, inpt[1:]):
            idx1, idx2 = stoi[str1], stoi[str2]
            bigram[idx1, idx2] += 1

    return bigram


if __name__ == "__main__":
    inputs = read_inputs(Path("names.txt"))
    stoi, itos = build_vocab(inputs)
    bigram = build_bigram_model(inputs, stoi)
    print(bigram)
