---
layout: page
parent: Lectures  
title: Overview of Large Language Models
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

# What is Language?

Language is a systematic means of communicating ideas or feelings using conventionalized signs, sounds, gestures, or marks.

## Text in Language

Text represents the written form of language, converting speech and meaning into visual symbols. Key aspects include:

### Basic Units of Text

Text can be broken down into hierarchical units:
- Characters: The smallest meaningful units in writing systems
- Words: Combinations of characters that carry meaning
- Sentences: Groups of words expressing complete thoughts
- Paragraphs: Collections of related sentences
- Documents: Complete texts serving a specific purpose

### Text Properties

Text demonstrates several key properties:
- Linearity: Written symbols appear in sequence
- Discreteness: Clear boundaries between units
- Conventionality: Agreed-upon meanings within a language community
- Structure: Follows grammatical and syntactic rules
- Context: Meaning often depends on surrounding text

Based on the above properties shared by different langauges, the NLP researchers develop a unified Machine Learning technique to model language data -- Large Language Models. Let's start to learn this unfied language modeling technique.

![Words in documents that get filtered out of C4](../images/c4-excluded.png)

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

# The Transformer Model: Revolutionizing Language Models

The Transformer model is a deep learning architecture that has revolutionized the field of natural language processing (NLP). It is based on a self-attention mechanism that allows it to handle long-range dependencies in text data, making it well-suited for tasks such as language translation, summarization, and question-answering.

The Transformer model consists of several key components that work together to process text data. In this article, we will focus on five of these components:

## Tokenizer: Turns words into tokens 

The tokenizer is the first component in the Transformer model pipeline. It converts raw text data into tokens, which are individual words or sub-words. Tokenization is a crucial step in NLP, as it allows us to process text data using mathematical operations.

For example, if the sentence is "Write a story", then the 4 corresponding tokens will be `<Write>`, `<a>`, `<story>`, and `<.>`.

## Embedding: Turns tokens into numbers (vectors)

Once the text data has been tokenized, the next step is to convert the tokens into numeric vectors that can be used as input to a deep learning model. This is done using an embedding layer, which maps each token to a high-dimensional vector in a continuous space. These embeddings capture the semantic meaning of the tokens and allow the model to understand the relationships between them.

In general embeddings send every word (token) to a long list of numbers. For example, if the sentence we are considering is "Write a story." and the tokens are `<Write>`, `<a>`, `<story>`, and `<.>`, then each one of these will be sent to a long vector, and we'll have four vectors.

## Positional Encoding: Adds order to the words in the text

The order of the words in a sentence is important for understanding the meaning of the text. To preserve this information, the Transformer model uses positional encoding, which adds a unique positional embedding to each token. These embeddings provide information about the relative position of each token in the sequence, allowing the model to distinguish between different positions in the text.

In the example, the vectors corresponding to the words "Write", "a", "story", and "." become the modified vectors that carry information about their position, labeled "Write (1)", "a (2)", "story (3)", and ". (4)".

## Transformer Block: Guesses the next word

The Transformer block is the heart of the Transformer model. It consists of two main sub-blocks: the attention block and the feedforward block. These blocks work together to guess the next word in the sequence based on the input tokens, embeddings, and positional encodings.

### Attention: Adds context to the text

The attention block in the Transformer model is responsible for adding context to the text data. It does this by attending to all the tokens in the input sequence and calculating a weighted sum of their embeddings. This weighted sum represents the context vector, which captures the most relevant information in the input sequence.

For example, consider these sentences to understand how context affects meaning:
- Sentence 1: The bank of the river.
- Sentence 2: Money in the bank.

Attention helps give context to each word, based on the other words in the sentence (or text).

### Feedforward

The feedforward block in the Transformer model takes the context vector generated by the attention block and passes it through a series of fully connected layers. These layers apply non-linear transformations to the context vector, allowing the model to make a prediction about the next word in the sequence.

## Softmax: Turns the scores into probabilities in order to sample the next word

Finally, the softmax function is used to turn the output of the feedforward block into a probability distribution over the vocabulary of possible next words. The model can then sample from this distribution to generate the most likely next word in the sequence.

## In conclusion

The Transformer model is a powerful deep learning architecture that has revolutionized the field of natural language processing. Its ability to handle long-range dependencies in text data makes it well-suited for a wide range of NLP tasks. The tokenizer, embedding, positional encoding, Transformer block, attention, feedforward, and softmax components work together to enable the model to guess the next word in a text sequence with remarkable accuracy. With continued advances in deep learning research, the Transformer model is poised to remain at the forefront of NLP for years to come.

# What are large language models?

Large language models are transformers with billions to trillions of parameters, trained on massive amounts of text data. These models have several distinguishing characteristics:

1. **Scale**: Models contain billions of parameters and are trained on hundreds of billions of tokens
2. **Architecture**: Based on the Transformer architecture with self-attention mechanisms
3. **Emergent abilities**: Complex capabilities that emerge with scale
4. **Few-shot learning**: Ability to adapt to new tasks with few examples