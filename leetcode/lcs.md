---
title: "Longest Common Subsequence"
date: 2024-11-19T12:13:56-05:00
draft: false
excerpt: Medium
---

## Problem Statement

Given two strings `text1` and `text2`, return the length of the longest common subsequence between the two strings if one exists, otherwise return `0`.

A subsequence is a sequence that can be derived from the given sequence by deleting some or no elements without changing the relative order of the remaining characters.

- For example, `"cat"` is a subsequence of `"crabt"`.

A common subsequence of two strings is a subsequence that exists in both strings.

Example 1:

```text
Input: text1 = "cat", text2 = "crabt" 
Output: 3 
```

Example 2:

```text
Input: text1 = "abcd", text2 = "abcd"
Output: 4
```

Example 3:

```text
Input: text1 = "abcd", text2 = "efgh"
Output: 0
```

Constraints:

- `1 <= text1.length, text2.length <= 1000`
- `text1` and `text2` consist of only lowercase English characters.

## Top-Down Solution

Recursive (times out):

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        if not text1 or not text2:
            return 0
        if text1[0] == text2[0]:
            return 1+self.longestCommonSubsequence(text1[1:], text2[1:])
        else:
            return max(self.longestCommonSubsequence(text1[1:], text2), self.longestCommonSubsequence(text1, text2[1:]))
```

Recursive using LRU Cache:

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        from functools import lru_cache

        @lru_cache(None)  # Built-in memoization
        def lcs(i: int, j: int) -> int:
            if i == len(text1) or j == len(text2):
                return 0
            if text1[i] == text2[j]:
                return 1 + lcs(i + 1, j + 1)
            else:
                return max(lcs(i + 1, j), lcs(i, j + 1))

        return lcs(0, 0)
```

## Bottom-Up Solution

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        dp = [[0 for _ in range(len(text2)+1)] for _ in range(len(text1)+1)]

        for i in range(len(text1)-1, -1, -1):
            for j in range(len(text2)-1, -1, -1):
                if text1[i] == text2[j]:
                    dp[i][j] = 1 + dp[i+1][j+1]
                else:
                    dp[i][j] = max(dp[i][j+1], dp[i+1][j])

        return dp[0][0]
```

## Notes

For the recursive approaches, perhaps using string lengths as function arguments to slice string instead of slicing string in recursive call would reduce space complexity.