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
&\mathcal{l} = \frac{1}{m} \sum_{i=1}^{m} -\log(A_{i}[y_i])
\end{aligned}
```




2. Backprop:

* As always, we super abuse notation, and simply refer to `dl/dW` as `dW` (maybe add better notation later on, include dims in decl, etc.)
* Assume we have `dA`, then we get the following:


```math
\boxed{dA_{ij} = -\frac{1}{m}\, \frac{\mathbf{1}\{j = y_i\}}{A_{ij}}.}
```

```math
\boxed{dZ = A \odot \Bigl( dA - \, \text{diag}(dA\, A^T)\,\mathbf{1}^T \Bigr)\,.}
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


- [ ] Backprop 
    - [x] Derive Backpropagation with your hands (I messed up the softmax for so long :/).
    - [x] Implement the derivations 
    - [ ] Verify with CPU code (just use o1 code) 

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

- [ ] MLP to MHA.. to flsh...?

Few small trips:
* Mini blog on softmax, compare to CuDNN.
* Profile your code, use NVIDIA NSIGHTs.

The ultimate goal is to code something interesting, e.g., flash attention. If not code then at least appreciate the intricacies of such high level implementations.

### dElEtE this:

* warp level primitives?
* check nvidia adds for keywords
* read some papers, related to prime intellect.


Jan 16:

- [ ] Better structure of the code 
    - [ ] add cuda error checks
    - [ ] just remove the .cuh?
    - [ ]  move every to src

- [ ] Backprop correctness
- [ ] Reduction and histogram
- [ ] Nsight Compute.
- [ ] Cache misses?
- [ ] Warp-level primitives
- [ ] Atomic level ops
Relevant for the logsoftmax loss backprop.
* `extern __shared__ ...` is used.
### Take this somewhere else:

* Dyn use of shared mem > `extern __shared__ ...`.

- A way to zero-init a matrix in the DRAM (so we can only update the ones that need to change)

```Cpp
float* d_matrix;
cudaMalloc((void**)&d_matrix, size * sizeof(float));
cudaMemset(d_matrix, 0, size * sizeof(float));
```
- when to use `cudaDeviceSynchronize();`, and when not to


- where to use cuda events / cuda streams? 

```Cpp
cudaStreamCreate(&stream);
mult<<<gridDim, blockDim, 0, stream>>>(/*...*/);
coalesced_bias<<<gridDims1, blockDims1, 0, stream>>>(/*...*/);
```

Instead of using `cudaDeviceSynchronize();`


* Quick thing regarding `cuDNN`, smth went off when my image was rest.

```bash
> wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
> rm /etc/apt/sources.list.d/cuda.list
> apt-get update
> apt-get -y install cudnn9-cuda-12
```
