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

### Core Insight
From a classification perspective, the number of categories directly impacts the learning difficulty - more categories require exponentially more training data to achieve adequate coverage.

### Comparing Two Approaches

#### Joint Probability Approach
When modeling $$P(w_1,...,w_n)$$ directly:
- Needs to predict $$V^n$$ categories
- Requires seeing enough samples of each possible sentence
- Most long sequences may never appear in training data
- Makes learning practically impossible

#### Conditional Probability Approach
When modeling $$P(w_i|w_1,...,w_{i-1})$$:
- Only predicts $$V$$ categories at each step
- Each word position provides a training sample
- Same words in different contexts contribute learning signals
- Dramatically improves data efficiency

### Numerical Example
Consider a language model with:
- Vocabulary size $$V = 10,000$$
- Sequence length $$n = 5$$

Then:
- Joint probability: Must learn $$10,000^5$$ categories
- Conditional probability: Must learn $$10,000$$ categories at each step

### Why This Matters
1. Training Data Requirements
- More categories require more training examples
- Each category needs sufficient representation
- Data requirements grow exponentially with category count

2. Learning Efficiency
- Smaller category spaces are easier to model
- More efficient use of training data
- Each word occurrence contributes to learning

3. Statistical Coverage
- Impossible to see all possible sequences
- But possible to see all words in various contexts
- Makes learning feasible with finite training data

### Conclusion
The conditional probability formulation cleverly transforms an intractable large-scale classification problem into a series of manageable smaller classification problems. This is the fundamental reason why language models can learn effectively from finite training data.

## Real-world Application: Text Completion

### The Prefix-based Generation Task
In practical applications, we often:
- Have a fixed prefix of text
- Need to predict/generate the continuation
- Don't need to generate text from scratch

### Examples
1. Auto-completion
- Code completion in IDEs
- Search query suggestions
- Email text completion

2. Text Generation
- Story continuation
- Dialogue response generation
- Document completion

### Why Conditional Probability Helps
The formulation $$P(w_i|w_1,...,w_{i-1})$$ naturally fits this scenario because:
- We can directly condition on the given prefix
- No need to model the probability of the prefix itself
- Can focus computational resources on predicting what comes next

### Comparison with Joint Probability
The joint probability $$P(w_1,...,w_n)$$ would be less suitable because:
- Would need to model probability of the fixed prefix
- Wastes computation on already-known parts
- Doesn't directly give us what we want (continuation probability)

This alignment between the mathematical formulation and practical use cases is another key advantage of the conditional probability approach in language modeling.

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