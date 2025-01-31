* This is not supposed to prove that you can derive gradients and do backprop from scratch. 

* The main objective of this project is:
    * Get used to CUDA with C++.
    * Use as much NVIDIA ecosystem as possible.
    * The ultimate goal is to code something interesting  (e.g., flash attention).

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

* As always, we refer to `dl/dW` as `dW` (maybe add better notation later on, include dims in decl, etc.). We get the following:


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

* In our case, `y` is one-hot-encoded, `dZ` simplifies as follows:

```math
dZ_{ij} = \frac{1}{m}(A_{ij} - \mathbf{1}\{j = y_i\})
```

### TODO/PROGRESS:

- [x] Forward pass, with correctness:
    - [x] tiled mult
    - [x] Numerically stable (log)-softmax implementation
    - [x] Cross-entropy loss (++ reduction pattern)
    - [x] Check correctness with cuDNN.

- [x] Backprop 
    - [x] Derive Backpropagation with your hands (I messed up the softmax for so long :/).
    - [x] Implement the derivations 
    - [x] Verify with CPU code (just use o1 code) 

- [x] Add complexity (for the sake of using more compute):
    - [x] Add one layer 
    - [x] start comparing to pytorch/python
    - [X] Systematic way to transfer weights accross between pytorch and C++.
    - [X] +90% acc in 1 epoch


- [] Work on your under 2/4 epochs accuracy, baseline SGD: 91.36%/92.74%:
    - [x] Adam (92.94% / 94.48%)
    - [ ] AdamW
    - [ ] tf is this muon optimizer? soap? why are you so late to this?

- [ ] Optimize (now that you have more compute complexity to do smth):
    - [x] Make softmax of `(2048x1024)` under 30 μs
    - [x] warp level primitives, is it even useful? 1000% all day.
    - [x] use cuda primitives for math ops (any differences?) hell yes!
    - [ ] profile your code, know how to use nsight compute. (Oupsie, no priveleges for gpu counters) 
    - [ ] play with compiler options / precisions. / how about see some instructions u dumbfuck?


- [ ] What's next? MLP to MHA to flash-attn...?

Few small trips:
* Profile your code, use NVIDIA NSIGHTs.
[x] softmax trip (was a nice trip, `__shfl_down_sync()` all day.

### Take this somewhere else:

- using `nvprof` with `nsys`
    - `nsys nvprfo ./app` lol.

- Dyn use of shared mem > `extern __shared__ ...`.
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


* Quick thing regarding `cuDNN`, smth went off when compute pod and my image was rest.

```bash
> wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.1-1_all.deb
> dpkg -i cuda-keyring_1.1-1_all.deb
> rm /etc/apt/sources.list.d/cuda.list
> apt-get update
> apt-get -y install cudnn9-cuda-12
```



* Weird side effects of my kernel when I put `printf` on. wtf? (like it actually work when `printf` is on). From Claude:


### Root Cause
The printf statement actually does affect execution because:
1. printf forces thread serialization
2. Acts as implicit synchronization point
3. Affects thread scheduling and timing
4. Can change memory access patterns

### Solution
To debug without affecting execution:
```cuda-cpp
__global__ void db(float* db, float* dZ, int hidden_dim) {
    int row = threadIdx.x + blockDim.x*blockIdx.x;
    
    // Add explicit synchronization if needed
    __syncthreads();
    
    float interVal = 0.0f;
    #ifdef DEBUG
    // Separate array for debugging values
    __shared__ float debug_vals[5];
    #endif
    
    for (unsigned int i = 0; i < BATCH_SIZE; i++) {
        float curr_val = dZ[i*hidden_dim+row];
        interVal += curr_val;
    }
    
    db[row] = interVal;
}
```

Better debugging approach:
1. Use CUDA debugger (cuda-gdb)
2. Store debug values in separate buffer
3. Add explicit synchronization points
4. Use CUDA events for timing analysis


### Debugging kernels is no easy task!

