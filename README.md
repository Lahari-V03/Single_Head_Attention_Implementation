# Single-Head Self-Attention Implementation

A step-by-step Jupyter notebook that builds a single-head self-attention mechanism from scratch using PyTorch.

1. **Open the notebook** in Jupyter/Colab
2. **Run all cells** to see the attention mechanism in action
3. **Examine the attention weights** to understand token relationships

This notebook walks through the **complete forward pass** of single-head self-attention:

1. **Input creation** - Batch of token embeddings
2. **QKV projection** - Linear transformations to query/key/value vectors  
3. **Attention scores** - Dot product similarity between tokens
4. **Softmax weights** - Normalized attention probabilities
5. **Output computation** - Weighted combination of value vectors

## 📚 Prerequisites & Background Reading

To fully understand this single-head attention implementation, start with these fundamentals:

### 1. Neural Networks Basics
**Interactive tool to visualize how neural networks learn:**
- [TensorFlow Playground](https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=spiral&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=40&networkShape=4,2&seed=0.05285&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false)

**What it teaches you:**
- How layers transform inputs through weights and activations
- How networks learn patterns from data

### 2. Transformers & Attention Theory
**Primary paper:**
- ['Attention is All You Need'](https://arxiv.org/abs/1706.03762) **→ Read Section 3.1**
  - **Scaled Dot-Product Attention**
  - **Q, K, V matrices** and their role
  - **Self-attention** vs encoder-decoder attention
  - Why scaling by \(\sqrt{d_k}\) matters

### 3. PyTorch Fundamentals
**Official PyTorch documentation:**
- [PyTorch Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) ← **Start here!**
  - Tensors (like NumPy arrays + GPU support)
  - Autograd (automatic differentiation) 
  - nn.Module (building neural networks)
  - Key operations: `@` (matrix multiply), `softmax()`, `transpose()`
 
## 📚 References

### Academic References
- **"Attention Is All You Need"** (Vaswani et al., 2017)  
  [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)  
  *This paper introduced the Transformer architecture. Section 3.1 explains scaled dot-product attention.*  
  **© Authors / NeurIPS 2017** - Cited for educational purposes.

### Tools & Libraries
- **PyTorch** (Paszke et al., 2019)  
  [Official Documentation](https://docs.pytorch.org/tutorials/beginner/basics/intro.html)  
  *BSD License - Free for commercial and research use.*  
  **© Meta Platforms, Inc. and affiliates**

- **TensorFlow Playground**  
  [playground.tensorflow.org](https://playground.tensorflow.org/)  
  *Interactive neural network visualization.*  
  **© Google**

### Notebook License
