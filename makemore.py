from collections import defaultdict


START = "<S>"
END = "<E>"

names = open("names.txt").read().splitlines()
bigrams = defaultdict(int)

for name in names:
    tokens = [START] + list(name) + [END]
    
    for t1, t2 in zip(tokens, tokens[1:]):
        bigrams[(t1, t2)] += 1

print(bigrams)