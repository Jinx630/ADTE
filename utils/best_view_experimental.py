import torch
import itertools


def combination(n, m):
    numbers = torch.arange(1, n + 1)
    all_combinations = list(itertools.combinations(numbers.tolist(), m))
    all_combinations = torch.tensor(all_combinations)
    return all_combinations

if __name__ == '__main__':
    n = 32
    m = 3
    result = combination(n, m)
    print(result.shape)