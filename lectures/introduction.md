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

## What is a language model?

A language model is a probability distribution over sequences of tokens. Given a vocabulary $$\mathcal{V}$$ of tokens, a language model $$p$$ assigns each sequence of tokens $$x_1,\dots,x_L \in \mathcal{V}$$ a probability:

$$p(x_1,\dots,x_L)$$

This probability represents how likely the sequence is according to the model's training. For example, with a vocabulary $$\mathcal{V} = \{\nl{the}, \nl{cat}, \nl{sat}, \nl{on}, \nl{mat}\}$$, the model might assign:

$$p(\nl{the}, \nl{cat}, \nl{sat}, \nl{on}, \nl{the}, \nl{mat}) = 0.01$$

$$p(\nl{cat}, \nl{the}, \nl{mat}, \nl{on}, \nl{sat}) = 0.0001$$ 

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