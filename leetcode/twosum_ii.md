---
title: "Two Integer Sum II"
date: 2024-11-13T10:46:42-05:00
draft: false
excerpt: Medium
---

## Problem Statement

Given an array of integers `numbers` that is sorted in non-decreasing order.

Return the indices (1-indexed) of two numbers, `[index1, index2]`, such that they add up to a given target number `target` and `index1 < index2`. Note that `index1` and `index2` cannot be equal, therefore you may not use the same element twice.

There will always be exactly one valid solution.

Your solution must use $O(1)$ additional space.

Example 1:

```
Input: numbers = [1,2,3,4], target = 3
Output: [1,2]
```
Explanation:
The sum of `1` and `2` is `3`. Since we are assuming a 1-indexed array, `index1 = 1`, `index2 = 2`. We return `[1, 2]`.


Constraints:

- `2 <= numbers.length <= 1000`
- `-1000 <= numbers[i] <= 1000`
- `-1000 <= target <= 1000`

## Naive Solution

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        i = 0; j = 1;
        while (i <= len(numbers)-2):
            cur = numbers[i]
            while (j <= len(numbers)-1):
                print(i,j)
                if cur + numbers[j] == target:
                    return [i+1,j+1]
                j += 1
            j = 1
            i += 1
```

When I didn't realize the list was sorted...

## Better Solution

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        i = 0; j = len(numbers)-1;
        while (i < j):
            if (guess := numbers[i] + numbers[j]) == target:
                return [i+1,j+1]
            elif guess > target:
                j -= 1;
            else: # guess < target
                i += 1;
```

When I did realize the list was sorted...