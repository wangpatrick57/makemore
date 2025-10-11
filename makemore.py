from pathlib import Path

import torch


SEP = "."


def read_examples(dataset_path: Path) -> list[str]:
    """
    The path should point to a file where each line contains an example of what we want
    to generate.
    """
    return open(dataset_path).read().splitlines()


def build_vocab(examples: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """
    Returns a mapping from each unique character that appears amongst the examples to a
    canonical token representing that character. Includes the separator character (SEP)
    as well, representing the start/end of an example. Returns the reverse mapping as
    the second return value.

    As a convention, we sort the characters alphabetically and put SEP last.
    """
    chars = sorted(set("".join(examples)))
    enc = {s: tok for tok, s in enumerate([SEP] + chars)}
    dec = {tok: s for s, tok in enc.items()}
    return enc, dec


def encode(example: str, enc: dict[str, int]) -> list[int]:
    """
    Encodes an example as a list of tokens with SEP at the start and the end.
    """
    return [enc[SEP]] + [enc[s] for s in example] + [enc[SEP]]


def decode(tokens: list[int], dec: dict[int, str]) -> str:
    """
    Decodes a list of tokens (that begin and end with SEP) into an example.
    """
    assert dec[tokens[0]] == SEP and dec[tokens[-1]] == SEP
    return "".join(dec[token] for token in tokens[1:-1])


def build_bigram_model(examples: list[str], enc: dict[str, int]) -> torch.Tensor:
    """
    Returns the bigram model, which is an NxN Tensor where N is the vocab size. Each
    row (i, j) of the Tensor is a probability distribution representing the probability
    of j appearing after i amongst the examples.
    """
    bigram = torch.zeros((len(enc), len(enc)), dtype=torch.float32)

    for example in examples:
        tokens = encode(example, enc)

        for tok1, tok2 in zip(tokens, tokens[1:]):
            bigram[tok1, tok2] += 1

    bigram /= bigram.sum(dim=1, keepdim=True)
    return bigram


def generate_example(
    bigram: torch.Tensor,
    enc: dict[str, int],
    dec: dict[int, str],
    generator: torch.Generator | None = None,
) -> str:
    """
    Generate an example probabilistically using the bigram model. Starts with the SEP
    character and generates new characters based on the probability distribution for
    the most recently generated character in bigram. Stops when SEP is generated.
    """
    tokens = [enc[SEP]]

    while True:
        tokens.append(
            torch.multinomial(bigram[tokens[-1]], 1, generator=generator).item()
        )

        if tokens[-1] == enc[SEP]:
            return decode(tokens, dec)


if __name__ == "__main__":
    examples = read_examples(Path("names.txt"))
    enc, dec = build_vocab(examples)
    bigram = build_bigram_model(examples, enc)
    generator = torch.Generator().manual_seed(2147483647)

    for _ in range(10):
        print(generate_example(bigram, enc, dec, generator=generator))
