---
layout: page
parent: Lectures  
title: Introduction to Large Language Models
nav_order: 1
usemathjax: true
---

$$
\newcommand{\nl}[1]{\textsf{#1}}
\newcommand{\attention}{\text{Attention}}
$$

Welcome to EECS 224! This is a graduate course on understanding and developing **large language models (LLMs)**.

1. [What is a language model?](#what-is-a-language-model)
1. [What are large language models?](#what-are-large-language-models)
1. [Grading of This Course](#grading-of-this-course)
1. [In-Course Questions](#in-course-questions)

# What is a Language Model?

## Mathematical Definition

A language model is fundamentally a probability distribution over sequences of words or tokens. Mathematically, it can be expressed as:

$$P(w_1, w_2, ..., w_n) = \prod_i P(w_i|w_1, ..., w_{i-1})$$

where:
- $$w_1, w_2, ..., w_n$$ represents a sequence of words or tokens
- The conditional probability of word $$w_i$$ given all previous words is:

  $$P(w_i|w_1, ..., w_{i-1})$$

For practical implementation, this often takes the form:

$$P(w_t|context) = \text{softmax}(h(context) \cdot W)$$

where:
- Target word: $$w_t$$
- Context encoding function: $$h(context)$$
- Weight matrix: $$W$$
- softmax normalizes the output into probabilities

# Language Model Probability Distribution

A language model is mathematically expressed as a probability distribution over sequences of words or tokens. Let's break down this concept:

## Basic Formula
The core equation is:
```
P(w₁, w₂, ..., wₙ) = ∏ᵢ P(wᵢ|w₁, ..., wᵢ₋₁)
```

This formula applies the Chain Rule of probability to decompose the joint probability into a product of conditional probabilities.

## Detailed Explanation

### Key Components
- The left side `P(w₁, w₂, ..., wₙ)` represents the probability of the entire sequence occurring
- The right side breaks this down into the product of conditional probabilities for each word given its preceding context

### Real-world Example
Consider the sentence "I love eating apples". The probability would be broken down as:
- P(I)
- P(love|I)
- P(eating|I love)
- P(apples|I love eating)

## Advantages of This Approach

### 1. Intuitive Alignment
This mathematical representation aligns with how humans process language - we naturally predict each word based on the previous context rather than randomly combining words.

### 2. Computational Efficiency
While calculating the probability of an entire sentence directly would be challenging, predicting the next word given a context is more manageable.

### 3. Context Capture
This formulation allows the model to capture long-term dependencies in language, as each word's prediction is based on all previous words in the sequence.

## Conclusion
This isn't simply a factorial multiplication, but rather a product of conditional probabilities. This mathematical expression accurately describes the language generation process: each word is predicted based on the words that came before it.

This approach forms the foundation of modern language models, enabling them to generate coherent and contextually appropriate text by learning these probability distributions from training data.

## Types of Language Models

1. **Statistical Language Models (SLM)**
   - Based on n-gram probability:
   
   $$P(w_n|w_1...w_{n-1}) \approx P(w_n|w_{n-k}...w_{n-1})$$
   
   - Limited by the Markov assumption

2. **Neural Language Models (NLM)**
   - Uses neural networks to learn

   $$P(w_t|context)$$
   
   - Can handle longer dependencies
   - Examples: LSTM, GRU based models

3. **Transformer-based Models**
   - Uses attention mechanism:
   
   $$\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d}})V$$
   
   - Where Q (Query), K (Key), V (Value) are learned matrices

## What are large language models?

Large language models are neural networks with billions to trillions of parameters, trained on massive amounts of text data. These models have several distinguishing characteristics:

1. **Scale**: Models contain billions of parameters and are trained on hundreds of billions of tokens
2. **Architecture**: Based on the Transformer architecture with self-attention mechanisms
3. **Emergent abilities**: Complex capabilities that emerge with scale
4. **Few-shot learning**: Ability to adapt to new tasks with few examples

## Grading of This Course

The course grade will be determined by the following components:

1. **In-Course Questions** (20%)
   - Regular questions during lectures
   - Participation and engagement
   - Short concept checks

2. **Programming Assignments** (30%)
   - 3 programming assignments throughout the quarter
   - Implementation of key concepts
   - Due every 3 weeks

3. **Midterm Project** (20%)
   - Individual or group project
   - Implementation and analysis of a specific LLM component
   - Written report and code submission

4. **Final Project** (30%)
   - Group project (2-3 students)
   - Original research or implementation
   - Final presentation and paper
   - Code repository with documentation

## In-Course Questions

During lectures, we will have interactive questions to help reinforce key concepts and ensure understanding. These questions will cover:

1. **Theoretical concepts**:
   - Language model basics
   - Probability theory
   - Model architectures 

2. **Implementation details**:
   - Code structure
   - Algorithm design
   - Optimization techniques

3. **Analysis and discussion**:
   - Model behavior 
   - Design choices
   - Performance trade-offs

Questions will be posted during class and students should be prepared to participate in discussions.