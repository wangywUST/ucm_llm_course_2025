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

# Tokenization in Language Models: Bridging Natural Language and Machine Understanding

## Overview

This chapter explores the fundamental role of tokenization in language models, examining how raw text is transformed into a format that machines can process effectively. As the bridge between human language and machine understanding, tokenization is crucial for the performance of modern language models.

## Foundations of Tokenization

### Core Concepts

A tokenizer transforms human-readable text into machine-processable tokens. This transformation involves:
- Converting continuous text into discrete units
- Mapping these units to numerical representations
- Managing a finite vocabulary for infinite language possibilities

### Mathematical Framework

The tokenization process supports the probabilistic nature of language models:

$$P(w_1, w_2, ..., w_n) = \prod_i P(w_i|w_1, ..., w_{i-1})$$

Where $$w_i$$ represents a token from the model's vocabulary. For practical implementation, we often use:

$$P(w_t|context) = \text{softmax}(h(context) \cdot W)$$

Where:
- Target word: $$w_t$$
- Context encoding function: $$h(context)$$
- Weight matrix: $$W$$

## Vocabulary Construction

### Algorithms

#### Byte Pair Encoding (BPE)

BPE is an iterative approach to vocabulary construction that starts with character-level tokens and progressively merges the most frequent pairs. This algorithm is particularly effective for handling rare words and morphologically rich languages.

```python
def train_bpe(texts, vocab_size):
    # Initialize vocabulary with characters
    vocab = set(''.join(texts))
    
    while len(vocab) < vocab_size:
        # Find most frequent pair
        pairs = get_most_frequent_pair(texts)
        if not pairs:
            break
            
        # Merge most frequent pair
        most_freq = max(pairs, key=pairs.get)
        vocab.add(''.join(most_freq))
        
        # Update texts with merged pair
        texts = [merge_pair(text, most_freq) for text in texts]
    
    return vocab
```

#### WordPiece

WordPiece enhances BPE by incorporating linguistic considerations and using a probability-based scoring mechanism for merging decisions:

```python
class WordPieceTokenizer:
    def __init__(self, vocab_size=30000):
        self.vocab_size = vocab_size
        self.vocab = set()
        
    def train(self, texts):
        # Initialize with characters
        self.vocab = self._get_base_vocab(texts)
        
        while len(self.vocab) < self.vocab_size:
            # Find best merge based on likelihood
            best_score = float('-inf')
            best_pair = None
            
            for pair in self._get_pairs(texts):
                score = self._calculate_score(pair, texts)
                if score > best_score:
                    best_score = score
                    best_pair = pair
                    
            if not best_pair:
                break
                
            # Add merged token to vocabulary
            self.vocab.add(''.join(best_pair))
```

### Probabilistic Analysis

The effectiveness of vocabulary size can be measured using information theory:

$$H(V) = -\sum_{i=1}^{|V|} p(w_i) \log p(w_i)$$

Where:
- $$H(V)$$ is the entropy of vocabulary $$V$$
- $$p(w_i)$$ is the probability of token $$w_i$$
- $$|V|$$ is the vocabulary size

### Optimization Function

The optimal vocabulary size can be found by minimizing:

$$L(V) = H(V) + \lambda|V|$$

Where $$\lambda$$ is a regularization parameter balancing vocabulary size and entropy.

## Text to Token Conversion

The process of converting text into tokens is a crucial step that follows vocabulary construction. This process involves several key components and considerations.

### Tokenization Process

#### Preprocessing

Text preprocessing ensures consistent input quality and includes several essential steps:
- Text normalization (case standardization)
- Special character handling
- Whitespace normalization
- Punctuation processing

Here's a basic implementation:

```python
def preprocess_text(text):
    # Lowercase normalization
    text = text.lower()
    # Handle special characters
    text = handle_special_chars(text)
    # Handle whitespace
    text = normalize_whitespace(text)
    return text
```

#### Token Lookup Strategies

After preprocessing, the text needs to be segmented into tokens using appropriate strategies.

##### Greedy Longest Match

This algorithm attempts to find the longest possible match in the vocabulary at each step. Benefits include:
- Maintains word integrity
- Reduces total token count
- Avoids oversegmentation

```python
def greedy_tokenize(text, vocab):
    tokens = []
    while text:
        # Try to find longest matching token
        longest_match = None
        for i in range(len(text), 0, -1):
            if text[:i] in vocab:
                longest_match = text[:i]
                break
        
        if longest_match:
            tokens.append(longest_match)
            text = text[len(longest_match):]
        else:
            # Handle unknown tokens
            tokens.append("[UNK]")
            text = text[1:]
    
    return tokens
```

##### Forward Maximum Matching

This approach uses a maximum length window to improve efficiency:
- Controls search space
- Improves processing speed
- Suitable for long texts

```python
def forward_tokenize(text, vocab, max_token_length=100):
    tokens = []
    start = 0
    
    while start < len(text):
        end = min(start + max_token_length, len(text))
        found = False
        
        while end > start:
            token = text[start:end]
            if token in vocab:
                tokens.append(token)
                start = end
                found = True
                break
            end -= 1
            
        if not found:
            tokens.append("[UNK]")
            start += 1
            
    return tokens
```

### Special Token Handling

Special tokens play crucial roles in modern language models, carrying specific functions and semantic information.

#### Common Special Tokens

Each special token serves a specific purpose:
- `[PAD]`: Used for sequence length alignment in batch processing
- `[UNK]`: Handles out-of-vocabulary words
- `[CLS]`: Special marker for classification tasks
- `[SEP]`: Separates different sentences or passages
- `[MASK]`: Used in masked language model training

Complete tokenizer implementation with special token handling:

```python
class Tokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.special_tokens = {
            "[PAD]": 0,
            "[UNK]": 1,
            "[CLS]": 2,
            "[SEP]": 3,
            "[MASK]": 4
        }
        
    def encode(self, text, max_length=None):
        # Preprocess text
        text = preprocess_text(text)
        
        # Tokenize
        tokens = self.greedy_tokenize(text)
        
        # Convert to IDs and add special tokens
        ids = [self.special_tokens["[CLS]"]]
        ids.extend([self.vocab.get(token, self.special_tokens["[UNK]"]) 
                   for token in tokens])
        ids.append(self.special_tokens["[SEP]"])
        
        # Handle max length
        if max_length:
            if len(ids) > max_length:
                ids = ids[:max_length-1] + [self.special_tokens["[SEP]"]]
            else:
                ids.extend([self.special_tokens["[PAD]"]] * 
                          (max_length - len(ids)))
        
        return ids
```

### Batch Processing

In practical applications, texts are usually processed in batches, requiring sequence length alignment and attention masks.

#### Padding and Attention Masks

Batch processing involves two key steps:
1. Sequence length unification (padding)
2. Attention mask generation (marking valid tokens)

```python
def prepare_batch(texts, tokenizer, max_length):
    # Tokenize all texts
    token_ids = [tokenizer.encode(text, max_length) for text in texts]
    
    # Create attention masks
    attention_masks = [
        [1 if token_id != tokenizer.special_tokens["[PAD]"] else 0 
         for token_id in ids]
        for ids in token_ids
    ]
    
    return {
        "input_ids": token_ids,
        "attention_mask": attention_masks
    }
```

### Decoding Process

The decoding process converts token sequences back to readable text, with special consideration for subword handling.

#### Token to Text Conversion

Key considerations during decoding:
- Special token filtering
- Subword merging
- Whitespace handling

```python
def decode(self, token_ids):
    # Filter special tokens
    tokens = [self.id_to_token[id] for id in token_ids 
             if id not in self.special_tokens.values()]
    
    # Handle subwords
    text = ""
    for token in tokens:
        if token.startswith("##"):  # BERT-style subwords
            text += token[2:]
        else:
            text += " " + token
    
    return text.strip()
```

Different tokenizers use different subword marking conventions:
- BERT uses ## prefix
- GPT uses Ġ prefix
- SentencePiece uses _ prefix

## Vocabulary Considerations

### Benefits of Larger Vocabularies

**Semantic Preservation**
- Maintains word-level meaning
- Reduces fragmentation
- Preserves domain-specific terms

**Efficiency**
- Shorter token sequences
- Reduced processing overhead
- Better context utilization

### Challenges

**Resource Demands**
- Larger embedding matrices
- Increased memory usage
- Higher computational costs

**Learning Difficulties**
- Data sparsity issues
- Longer training time
- Potential overfitting

## Future Developments

Current research directions include:
- Adaptive tokenization methods
- Neural tokenizers
- Multilingual approaches
- Vocabulary compression techniques

## Conclusion

Tokenization serves as the crucial interface between human language and machine processing. Understanding its principles, from vocabulary construction to text conversion, enables better language model development and deployment. As the field evolves, new approaches and optimizations continue to emerge, driven by the increasing demands of modern language models.

## References

1. Sennrich, R., et al. (2016). Neural Machine Translation of Rare Words with Subword Units
2. Wu, Y., et al. (2016). Google's Neural Machine Translation System
3. Kudo, T. (2018). Subword Regularization
4. Vaswani, A., et al. (2017). Attention Is All You Need
5. Clark, K., et al. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

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