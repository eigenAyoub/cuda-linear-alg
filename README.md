* This is not supposed to prove that you can derive gradients and do backprop from scratch. 
* The main objective of this project is:
    * Get used to CUDA with C++.
    * Use as much NVIDIA ecosystem as possible.

### Steps:

1. Implement this minimal forward setting:

```math
\begin{aligned}
&Z = \underbrace{XW_1}_{Y} + b_1 \\
&A = \text{softmax}(Z) \\
&\mathcal{L} = \text{CrossEntropyLoss}(A, Y_{\text{true}}) 
= [-\log(A_{i}[y_{\text{true}_i}])]_{i=1}^{m} \\
&\mathcal{l} = \frac{1}{m} \sum_{i=1}^{m} -\log(A_{i}[y_{\text{true}_i}])
\end{aligned}
```
For each sample \(i\) and class \(j\), we define




2. Backprop:

* As always, we super abuse notation, and simply refer to `dl/dW` as `dW` (maybe add better notation later on, include dims in decl, etc.)
* Assume we have `dA`, then we get the following:


```math
\boxed{dA_{ij} = -\frac{1}{m}\, \frac{\mathbf{1}\{j = y_i\}}{A_{ij}}.}
```

```math
\boxed{dZ = A \odot \Bigl( dA - \, \operatorname{diag}(dA\, A^T)\,\mathbf{1}^T \Bigr)\,.}
```
```math
\boxed{dW = X^T dZ.}
```
```math
\boxed{db = \mathbf{1}^T dZ}
```

### TODO/PROGRESS:

- [x] Forward pass, with correctness:
    - [x] tiled mult
    - [x] Numerically stable (log)-softmax implementation
    - [x] Cross-entropy loss (++ reduction pattern)
    - [x] Check correctness with cuDNN.


- [X] Derive Backpropagation with your hands (I messed up the softmax for so long :/).

- [ ] Optimize the forward pass (later):
    - [ ] profile your code, know how to use nsight 
    - [ ] use cuda primitives for math ops (any differences?)
    - [ ] warp level primitives, is it even useful?
    - [ ] high batch_size and profile your code. 
    - [ ] compare / profile benchmark / have fun.
    - [ ] play with compiler options, precisions. 
    - [ ] ask Claude/o1 what's wrong with my code. 

- [ ] Improve training:
    - [ ] use Adam?
    - [ ] more layers, etc.

- [ ] Optimize, Optimize, Optimize.

- [ ] MLP to Att.. to flsh...?

Few small trips:
* Mini blog on softmax, compare to CuDNN.
* Profile your code, use NVIDIA NSIGHTs.

The ultimate goal is to code something interesting, e.g., flash attention. If not code then at least appreciate the intricacies of such high level implementations.

### dElEtE this:
* `extern __shared__ ...` is used.
* warp level primitives?
* check nvidia adds for keywords
* read some papers, related to prime intellect.

Jan 16:

[ ] Better structure of the code.
[ ] Backprop
[ ] Reduction and histogram
[ ] Nsight Compute.

[ ] Cache misses?
[ ] Warp-level primitives

[ ] A way to automatically init a matrix to 0's in the DRAM, so we can only update the ones that need to change
Relevant for the logsoftmax loss backprop.