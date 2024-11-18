---
title: "Is Palindrome"
date: 2024-11-12T17:29:10-05:00
draft: false
excerpt: Easy
---

## Problem Statement

Given a string `s`, return true if it is a palindrome, otherwise return false.

A palindrome is a string that reads the same forward and backward. It is also case-insensitive and ignores all non-alphanumeric characters.

Example 1:

```
Input: s = "Was it a car or a cat I saw?"
Output: true
```

Explanation: After considering only alphanumerical characters we have "wasitacaroracatisaw", which is a palindrome.

Example 2:

```
Input: s = "tab a cat"
Output: false
```
Explanation: "tabacat" is not a palindrome.

Constraints:

- `1 <= s.length <= 1000`
- `s` is made up of only printable ASCII characters.

## Solution

```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        s = re.sub(r'[^a-zA-Z0-9]','', s.lower())
        i = 0; j = len(s)-1;
        while(i <= j):
            if s[i] == s[j]:
                i+=1; j-=1;
            else:
                return False
        return True
```

## Notes

Super simple two pointers.
        