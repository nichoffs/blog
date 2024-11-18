---
title: "Binary Search"
date: 2024-11-13T13:42:34-05:00
draft: false
excerpt: Easy
---

## Problem Statement

You are given an array of distinct integers `nums`, sorted in ascending order, and an integer `target`.

Implement a function to search for `target` within `nums`. If it exists, then return its index, otherwise, return `-1`.

Your solution must run in $O(\log n)$ time.

<ins>Example 1</ins>:
```text
Input: nums = [-1,0,2,4,6,8], target = 4
Output: 3
```

<ins>Example 2</ins>:
```text
Input: nums = [-1,0,2,4,6,8], target = 3
Output: -1
```

## Solution

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        lo = 0; hi = len(nums)-1;
        while (lo <= hi):
            mid =  (lo + hi) // 2
            if nums[mid] > target:
                hi = mid - 1
            elif nums[mid] < target:
                lo = mid + 1
            else:
                return mid
        return -1
```    