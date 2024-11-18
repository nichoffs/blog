---
title: "Reverse Linked-List"
date: 2024-11-13T13:56:22-05:00
draft: false
excerpt: Easy
---

## Problem Statement

Given the beginning of a singly linked list `head`, reverse the list, and return the new beginning of the list.

<ins>Example 1:</ins>

```text
Input: head = [0,1,2,3]
Output: [3,2,1,0]
```

<ins>Example 2:</ins>

```text
Input: head = []
Output: []
```

## Solution

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next

class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, cur = None, head
        while cur is not None:
            # store next
            nxt = cur.next
            # reverse node
            cur.next = prev
            # increment pointers
            prev = cur
            cur = nxt
        return prev
```

## Notes

        