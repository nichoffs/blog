---
title: "What I Worked On This Summer"
date: 2024-11-10T15:35:25-05:00
draft: false
excerpt: GPTs, MNISTs, and C hell.
---

![San Sebastian](https://www.spain.info/.content/imagenes/cabeceras-grandes/pais-vasco/san-sebastian-vista-aerea-s628199351.jpg)

I spent this summer in the beautiful, delicious, friendly [San Sebastian, Spain](https://en.wikipedia.org/wiki/San_Sebastián). While working remotely from the States, I took on some passion projects geared at enhancing both my deep-learning foundations and practical experience. In this article, I'll explain what those projects were. 

Sidenote: If you are looking to vacation in Southern Europe, go to San Sebastian -- enough said.

## TinyStories GPT2

Check out the source code [here](https://github.com/nichoffs/gpt2).

Upon the release of [Andrej Karpathy's 4 hour GPT2 implementation and pre-training video](https://www.youtube.com/watch?v=l8pRSuU81PU), I decided to reimplement a smaller version of my own based on the [TinyStories](https://arxiv.org/pdf/2305.07759) dataset. Instead of PyTorch, I used TinyGrad (and loved it). 

My goal with this was to downscale enough such that single-GPU (A100), single-day (~6 hours) training was possible while still having relatively coherent generations. Like Karpathy, I trained the 124M parameter model. As expected, my model had much less diverse generations than the original model (I trained on a significantly more limited dataset). Regardless, I was really impressed with the result. 

Check out the output below and note that "Once upon a time" was the prompt.

<ins>**Pre-trained Generation**</ins>
```text
Once upon a time it was all about a team of robots or robots and I would have gotten a better joke out of them than this, and when it came to the actual robot, I had no idea what the whole plan was about.
```

<ins>**Checkpointed Generation**</ins>
```text
Once upon a time, there was a girl called Kitty. Grace had an exciting time. One day, Blue was playing in the park. She had a pretty coat and a picture full of a big blue ball. It was an unusual day, and
```

## MNIST4All