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

# Why Use Conditional Probability in Language Models?

The fundamental reason for expressing language models using conditional probability:

$$P(w_1, w_2, ..., w_n) = \prod_i P(w_i|w_1, ..., w_{i-1})$$

lies in the reduction of prediction space.

## Comparing the Prediction Spaces

### Joint Probability Approach
When directly predicting P(w₁,...,wₙ):
- With vocabulary size V
- Need to predict V^n possible combinations
- Results in an exponential prediction space

### Conditional Probability Approach
When using conditional decomposition:
- Only need to predict one word at each step
- Each prediction considers V possibilities
- Total prediction space reduces from V^n to n×V

## Key Insight
This dramatic reduction in prediction space is what makes neural network approximation feasible. All other claimed benefits (like intuitive alignment, computational efficiency, or context capture) are not actual advantages of the conditional probability formulation.

The ability to constrain each prediction step to a manageable vocabulary-sized space is the core reason why language models are structured this way.

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