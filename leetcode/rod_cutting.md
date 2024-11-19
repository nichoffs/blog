---
title: "Rod Cutting Problem"
date: 2024-11-19T15:14:45-05:00
draft: false
excerpt: Not on LeetCode 
---

## Problem Statement

You are given a rod of length `n` and an array prices of size `n`, where `prices[i]` represents the price of a rod of length `i+1`. You want to cut the rod into pieces of various lengths (or leave it uncut) so that the total price is maximized.

Return the maximum profit you can obtain after accounting for all cuts.

```python
def rodCutting(prices: List[int], n: int) -> int:
```

Input:
- `prices` (`List[int]`): An array of integers where `prices[i]` is the price of a rod of length `i+1`.
- `n` (`int`): The length of the rod.

Output:
`int`: The maximum profit you can achieve.

Constraints:
- `1 <= n <= 100`
- `1 <= prices[i] <= 1000`

## Solution

```python
prices = [1,5,8,9,10,17,17,20,24,30]
```

# Recursive

```python
def cut_rod(prices, n):
    if n == 0:
        return 0
    q = float('-inf')
    for i in range(n):
       q = max(q, prices[i] + cut_rod(prices, n-(i+1)))
    return q

cut_rod(prices, len(prices))
```
```text
30
```

# Recursive memoized with list

```python
def cut_rod(prices, n, memo=None):
    if memo is None: memo = [float("-inf") for _ in range (n)]
    if n == 0: return 0
    if memo[n-1] != float("-inf"): return memo[n-1]
    
    q = float('-inf')
    for i in range(n):
        q = max(q, prices[i] + cut_rod(prices, n - (i + 1), memo))
    
    memo[n-1] = q
    return q

cut_rod(prices, len(prices))
```
```text
30
```

# Recursive memoized with dictionary

```python
def cut_rod(prices, n, memo=None):
    if memo is None: memo = {}
    if n == 0:return 0
    if n in memo: return memo[n]
    
    q = float('-inf')
    for i in range(n):
        q = max(q, prices[i] + cut_rod(prices, n - (i + 1), memo))
    
    memo[n] = q
    return q

cut_rod(prices, len(prices))
```
```text
30
```

# Bottom-Up

```python
def cut_rod(prices, n):
    memo = [0] * (n + 1)
    for i in range(1, n + 1): 
        q = float("-inf")
        for j in range(i):  
            q = max(q, prices[j] + memo[i - (j + 1)])
        memo[i] = q  
    return memo[n]

cut_rod(prices, len(prices))
```
```text
30
```