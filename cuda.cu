#include "utility.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include "numgen.c"

__global__ void primeCount(unsigned long int* numbers, int* primes, long size) 
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (index < size) 
    {
        unsigned long int number = numbers[index];
        
        if (number < 2) 
            primes[index] = 0;
        
        else 
        {
            primes[index] = 1;
            for (unsigned long int i = 2; i * i <= number; i++) 
            {
                if (number % i == 0) 
                {
                    primes[index] = 0;
                    break;
                }
            }
        }
    }
}

int main(int argc,char **argv) 
{
    Args ins__args;
    parseArgs(&ins__args, &argc, argv);

    //program input argument
    long inputArgument = ins__args.arg; 
    unsigned long int *numbers = (unsigned long int*)malloc(inputArgument * sizeof(unsigned long int));
    numgen(inputArgument, numbers);

    struct timeval ins__tstart, ins__tstop;
    gettimeofday(&ins__tstart, NULL);

    // allocate memory
    unsigned long int *device_numbers;
    int *device_primes;
    
    cudaMalloc(&device_numbers, inputArgument * sizeof(unsigned long int));
    cudaMalloc(&device_primes, inputArgument * sizeof(int));

    // numbers to device
    cudaMemcpy(device_numbers, numbers, inputArgument * sizeof(unsigned long int), cudaMemcpyHostToDevice);

    // run your CUDA kernel(s) here
    int blockSize = 256;
    int numBlocks = (inputArgument + blockSize - 1) / blockSize;
    primeCount<<<numBlocks, blockSize>>>(device_numbers, device_primes, inputArgument);
	
    /* 
    for(long i=0;i<inputArgument;i++)
    	printf("%ld\n",numbers[i]);
    */
    
    // primes to host
    int *primes = (int*)malloc(inputArgument * sizeof(int));
    cudaMemcpy(primes, device_primes, inputArgument * sizeof(int), cudaMemcpyDeviceToHost);

    // count
    int primeCount = 0;
    
    for (long i = 0; i < inputArgument; i++) 
        primeCount += primes[i];

    printf("Number of primes: %d\n", primeCount);

    // synchronize/finalize your CUDA computations
    cudaDeviceSynchronize();

    gettimeofday(&ins__tstop, NULL);
    ins__printtime(&ins__tstart, &ins__tstop, ins__args.marker);

    // free memory
    free(numbers);
    free(primes);
    cudaFree(device_numbers);
    cudaFree(device_primes);
}

