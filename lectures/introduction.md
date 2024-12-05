---
layout: page
parent: Lectures  
title: Introduction to Large Language Models
nav_order: 1
usemathjax: true
---

$$
\newcommand{\bx}{\mathbf{x}}
\newcommand{\by}{\mathbf{y}}
\newcommand{\bz}{\mathbf{z}}
\newcommand{\bW}{\mathbf{W}}
\newcommand{\bh}{\mathbf{h}}
\newcommand{\attention}{\text{Attention}}
\newcommand{\nl}[1]{\textsf{#1}}
$$

Welcome to EECS 224! This is a graduate course on understanding and developing **large language models (LLMs)**.

1. [What are large language models?](#what-are-large-language-models)
1. [Mathematical foundations](#mathematical-foundations)
1. [Core architectures](#core-architectures)
1. [Training methodology](#training-methodology)

## What are large language models?

The fundamental definition of a large language model is a neural network that can process and generate text by modeling the probability distribution over sequences of tokens. Given a vocabulary $$\mathcal{V}$$ of tokens, an LLM $$p$$ assigns each sequence of tokens $$x_1,\dots,x_L \in \mathcal{V}$$ a probability:

$$p(x_1,\dots,x_L)$$

This probability represents how likely the sequence is according to the model's training. For example, with a vocabulary $$\mathcal{V} = \{\nl{the}, \nl{cat}, \nl{sat}, \nl{on}, \nl{mat}\}$$, the model might assign:

$$p(\nl{the}, \nl{cat}, \nl{sat}, \nl{on}, \nl{the}, \nl{mat}) = 0.01$$

$$p(\nl{cat}, \nl{the}, \nl{mat}, \nl{on}, \nl{sat}) = 0.0001$$ 

The second sequence gets lower probability because it's ungrammatical, demonstrating how LLMs implicitly learn **syntactic knowledge**.

### Autoregressive modeling

The joint probability $$p(x_{1:L})$$ is typically factored using the chain rule:

$$p(x_{1:L}) = \prod_{i=1}^L p(x_i|x_{1:i-1})$$

For example:
```python
def compute_probability(model, tokens):
    prob = 1.0
    for i in range(len(tokens)):
        prob *= model.get_conditional_prob(tokens[i], tokens[:i])
    return prob
```

Each term $$p(x_i|x_{1:i-1})$$ represents the probability of token $$x_i$$ given all previous tokens. This autoregressive factorization enables both:

1. **Training**: We can maximize the likelihood of the training data
2. **Generation**: We can sample tokens one at a time conditioned on previous tokens

## Mathematical foundations

The key mathematical concepts underlying LLMs include:

### Self-attention mechanism

The core building block is self-attention, which computes weighted combinations of value vectors $$\bv$$ based on query-key compatibility:

$$\attention(\bQ, \bK, \bV) = \text{softmax}\left(\frac{\bQ\bK^\top}{\sqrt{d_k}}\right)\bV$$

Where:
- $$\bQ \in \mathbb{R}^{n \times d_k}$$ contains query vectors
- $$\bK \in \mathbb{R}^{n \times d_k}$$ contains key vectors  
- $$\bV \in \mathbb{R}^{n \times d_v}$$ contains value vectors
- $$d_k$$ is the dimension of the key/query vectors
- $$n$$ is the sequence length

The scaling factor $$\sqrt{d_k}$$ prevents the dot products from growing too large in magnitude.

### Feed-forward layers

Between attention layers, we have position-wise feed-forward networks:

$$\text{FFN}(\bx) = \text{ReLU}(\bx\bW_1 + \mathbf{b}_1)\bW_2 + \mathbf{b}_2$$

This allows the model to process each position's representations independently.

## Core architectures 

Modern LLMs are based on the Transformer architecture, which consists of:

1. **Token embeddings**: Convert discrete tokens to vectors
2. **Positional embeddings**: Encode position information 
3. **Multiple layers** of:
   - Multi-head self-attention
   - Feed-forward networks
   - Layer normalization
4. **Output layer**: Projects to vocabulary probabilities

The basic structure looks like:

```python
class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, nhead) 
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)
```

## Training methodology

Training large language models involves several key components:

### Data preprocessing

The first step is tokenization - converting raw text into integer sequences. Common approaches include:

1. **Byte-pair encoding (BPE)**
2. **WordPiece**
3. **SentencePiece** 

For example, using BPE:
```python
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
text = "The quick brown fox"
tokens = tokenizer.encode(text)
# tokens: [464, 2159, 2829, 4062]
```

### Optimization

Training uses variants of stochastic gradient descent with:

- Large batch sizes (often thousands)
- Learning rate scheduling
- Gradient clipping
- Mixed precision training

The loss function is typically cross-entropy over next-token prediction:

$$\mathcal{L} = -\sum_{i=1}^L \log p(x_i|x_{1:i-1})$$

### Distributed training

Due to model size, training requires sophisticated parallelization:

1. **Data parallelism**: Split batches across devices
2. **Model parallelism**: Split model layers across devices
3. **Pipeline parallelism**: Different stages on different devices

## Summary

- LLMs are probability distributions over token sequences
- They use Transformer architectures with self-attention
- Training requires careful optimization and parallelization
- Core components include tokenization, embedding, attention

## Further reading

1. Vaswani et al. "Attention is All You Need"
2. Brown et al. "Language Models are Few-Shot Learners" 
3. Kaplan et al. "Scaling Laws for Neural Language Models"
4. Chowdhery et al. "PaLM: Scaling Language Modeling with Pathways"