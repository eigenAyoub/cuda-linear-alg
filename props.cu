#include <iostream>
#include <cuda_runtime.h>
#include <unistd.h>


int main(){

    int devCount;
    cudaGetDeviceCount(&devCount);

    std::cout << "Number of CUDA devices: " << devCount << std::endl;

    cudaDeviceProp devProp;

    auto bytesToKB = [](size_t b){ return static_cast<float>(b)/(1024);};
    auto bytesToGB = [](size_t b){ return static_cast<float>(b)/(1024*1024*1024);};

    for (int i = 0; i < devCount; i++){
        cudaGetDeviceProperties(&devProp, i);
        std::cout << "Device " << i << ": " << devProp.name << std::endl;
        std::cout << "Compute capability: " << devProp.major << "." << devProp.minor << std::endl;

        std::cout << "Total global memory: "     << bytesToGB(devProp.totalGlobalMem)
                                                 << " GB\n";
        std::cout << "Shared memory per block: " << bytesToKB(devProp.sharedMemPerBlock) 
                                                 << " KB\n";
        std::cout << "Shared memory per SM: "    << bytesToKB(devProp.sharedMemPerMultiprocessor) 
                                                 << " KB\n";

        std::cout << "Registers per block: " << devProp.regsPerBlock << std::endl;
        std::cout << "Warp size: " << devProp.warpSize << std::endl;
        std::cout << "Max threads per block: " << devProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max threads per SM: " << devProp.maxThreadsPerMultiProcessor  << std::endl;
        std::cout << "Max threads dimensions: " << devProp.maxThreadsDim[0] << " x " << devProp.maxThreadsDim[1] << " x " << devProp.maxThreadsDim[2] << std::endl;
        std::cout << "Max grid size: " << devProp.maxGridSize[0] << " x " << devProp.maxGridSize[1] << " x " << devProp.maxGridSize[2] << std::endl;
        std::cout << "Clock rate: " << devProp.clockRate << std::endl;
        std::cout << "Total constant memory: " << devProp.totalConstMem << std::endl;
        std::cout << "Texture alignment: " << devProp.textureAlignment << std::endl;
        std::cout << "Multiprocessor count (#SMs): " << devProp.multiProcessorCount << std::endl;
    }
    return 0;
}
