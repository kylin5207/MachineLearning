from itertools import combinations, permutations
from typing import List


def compute_combination(numbers: List, k: int):
    """从给定样本中选择k个，计算各种组合的情况"""
    # 生成所有可能的组合
    all_combinations = list(combinations(numbers, k))
    return all_combinations

def compute_permutation(numbers: List, k: int):
    """从给定样本中选择k个，计算各种排列的情况"""
    all_permutations = list(permutations(numbers, 3))
    return all_permutations


numbers = range(3)
k = 3
all_permutations = compute_permutation(numbers, k)
print(all_permutations)