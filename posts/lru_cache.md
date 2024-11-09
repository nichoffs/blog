---
title: "Understanding TinyGrad: LRU Cache"
date: 2024-11-04T00:24:15-05:00
draft: false
excerpt: "My LRU data structure"
---

TinyGrad uses a least recently used (LRU) cache to avoid repeatedly allocating and freeing GPU memory. I didn't know what an LRU cache was before reading TinyGrad, so I figured I'd go through an implementation myself. An LRU cache is a type of cache that automatically discards the least recently accessed item when the cache reaches its fixed capacity.

So, when you "free" a buffer in TinyGrad, it actually gets stored in a cache, indexed by its size and options. Later, when you need a new buffer of the same size and options, instead of allocating fresh GPU memory, TinyGrad can reuse one of these "freed" buffers by simply overwriting its contents.

Traditional LRU caches (like those used in CPU caches or web browsers) often use a hash map + doubly linked list because they need constant time operations for finding an element (lookup in cache), removing the least recently used element (when capacity is reached), and moving an element to the front (when an item is looked up and becomes the most recently used). Hash maps offer $O(1)$ lookup into any point in the array and linked lists offer $O(1)$ insertion/removal at any point. 

While, at first, I implemented the LRU cache using the traditional method, I found [this article](https://www.geeksforgeeks.org/lru-cache-in-python-using-ordereddict/) that uses Python's `OrderDict`. It ended up being the exact thing I was looking for. I can pop the last accessed item, move items to "most recently accessed" once `get` is called, and insert new items.


```python
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def __repr__(self):
        return "Capacity: " + str(self.capacity) + " \nCache: " + str(self.cache)

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```


```python
cache = LRUCache(3)

cache.put("one", 1)
cache.put("two", 2)
cache.put("three", 3)

cache.put("four", 4) # should remove "one"
print(cache)
```
```text
Capacity: 3 
Cache: OrderedDict({'two': 2, 'three': 3, 'four': 4})
```


Here's the previous version using a doubly linked list and a hash map. I decided to convert it to the format specified by [this Leetcode problem](https://leetcode.com/problems/lru-cache/):


```python
class Node:
        def __init__(self, key=None, val=None):
            self.key = key
            self.val = val
            self.next = None
            self.prev = None

        def __repr__(self):
            return str(self.val)

class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.lru = Node()
        self.mru = Node()
        self.lru.next = self.mru
        self.mru.prev = self.lru

    def __repr__(self):
        return "Capacity: " + str(self.capacity) + " \nCache: " + str(self.cache)

    def get(self, key: int) -> int:
        if key in self.cache:
            node = self.cache[key]
            # make node mru because it was accessed
            self._remove_node(node)
            self._add_mru(node)

            return node.val
        else:
            return -1

    def _add_mru(self, node: Node):
        node.prev = self.mru.prev # make new mru's prev the old mru
        self.mru.prev.next = node # make old mru's next the new mru
        # make our node the new mru
        node.next = self.mru
        self.mru.prev = node

    # remove node from linked list and hash map
    def _remove_lru(self):
        self._remove_node(node := self.lru.next)
        del self.cache[node.key]
    
    # remove node from linked list
    def _remove_node(self, node):
        node.next.prev = node.prev
        node.prev.next = node.next

    def put(self, key: int, val: int) -> None:

        # if node is already present, update value and make mru
        if key in self.cache: 
            self._remove_node(node := self.cache[key]) # remove node
            self._add_mru(node) # add back node at mru position
            node.val = val # update value

        # if node isn't present, 
        #   1. remove lru if at capacity 
        #   2. add to hashmap
        #   3. set new node as mru
        else: 
            if len(self.cache) >= self.capacity:
                self._remove_lru()
            node = Node(key, val)
            # initialize in hash map and make new mru in linked list
            self.cache[key] = node
            self._add_mru(node)
```


```python
cache = LRUCache(3)

cache.put("one", 1)
cache.put("two", 2)
cache.put("three", 3)

cache.put("four", 4) # should remove "one"
print(cache)
```

```text
Capacity: 3 
Cache: {'two': 2, 'three': 3, 'four': 4}
```

testing railway