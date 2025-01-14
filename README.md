* This is not supposed to prove that you can derive gradients and do backprop from scratch. 
* The main objective of this project is:
    * Get used to CUDA with C++.
    * Use as much NVIDIA ecosystem as possible.


### Steps:


First we implement this minimal forward setting:

```math
\begin{aligned}
&Z = \underbrace{XW_1}_{Y} + b_1 \\
&A = \text{softmax}(Z) \\
&\mathcal{L} = \text{CrossEntropyLoss}(A, Y_{\text{true}}) 
= [-\log(A_{i}[y_{\text{true}_i}])]_{i=1}^{m} \\
&\mathcal{l} = \frac{1}{m} \sum_{i=1}^{m} -\log(A_{i}[y_{\text{true}_i}])
\end{aligned}
```

### TODO/PROGRESS:

- [ ] Forward pass, with correctness:
    - [x] tiled mult
    - [ ] Softmax implementation
    - [ ] Cross-entropy loss (++ reduction pattern)

- [ ] Backpropagation
- [ ] Add one intermediate layer.
- [ ] Optimize, Optimize, Optimize.
- [ ] MLP to ...?

Few small trips:
* Mini blog on softmax, compare to CuDNN.
* Profile your code, use NVIDIA NSIGHTs.

The ultimate goal is to code something interesting, e.g., flash attention. If not code then at least appreciate the intricacies of such high level implementations.

### Some stuff:

* When adding bias, no need for share memory.
    * But it's cool, you've seen a case where `extern __shared__ ...` is used.


### dElEtE this:
