---
title: Speedrunning a startup with FastHMTL and LLMs
date: 2024-10-21T18:33:10-04:00
draft: true
excerpt: What do I have to lose when CapEx is non-existent?
---

I've been inspired by [Pieter Levels](https://x.com/levelsio), a solo entrepreneur known for creating profitable, capital-light AI startups with a minimal tech stack (HTML/JS/PHP/[replicate.ai](https://replicate.com)). [PhotoAI](https://photoai.com) is his most succesful venture, returning about $157k/month according to his bio. The idea is dead simple -- people upload their photos, and he charges them to use a text-to-image model (fine-tuned on their portraits) to generate new images of themselves. I will use this idea as the foundation of a "startup".

## The perception of value

I get the sense that a lot of people (and by that, I mean the general population) still aren't truly aware of the quality of modern text-to-image models, particularly with the personalization factor added through fine-tuning. For those unfamiliar with the space, an image generator framed as a specially trained AI model for whatever niche they're into would be exciting and valuable. This is the basic premise of my marketing strategy. [PhotoAI](https://photoai.com) achieves a similar novelty, except it's leveraging the fact that people enjoy seeing photos of themselves (as opposed to photos related to some niche or community). 

I'm lazy, so before I go about fine-tuning my own image model, I'll test the premise using an already fine-tuned flux model, which has an inference API endpoint on replicate. Right now, I'm thinking [childrens book illustrations](https://replicate.com/samsa-ai/flux-childbook-illustration) for parents.

You may wonder whether this idea really warrants being made into a business. After all, it's not really providing novel value.

## Outlining the stack

I enjoy Python, but I hate bloat. In an ideal world, I could keep everything (meaning web and database) in Python. With FastHTML, I I can do it without feeling significantly constrained.