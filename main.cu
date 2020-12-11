#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NDIM 3
#define MAX 200

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
/**
 * CUDA Kernel Device code
 *
 */
extern "C" void write_output(char fname[MAX],int *in,unsigned long int nconf);

__global__ void int_generator (int *d_out,unsigned long int size);

void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true);

int main(int argc, char **argv)
{
    
  unsigned long int size;
  char fname[MAX];

  /* Cuda */
  cudaError_t err = cudaSuccess;
  cudaDeviceProp prop;
  int device;


  if (argc != 2){
    printf("Usage: main  <size of array> \n");
    return(0);
  }

  //Initialization

  size = atoi(argv[1]);

  /* Memory allocation */

  /* Host */ 
  int *h_num = (int*) calloc(size, sizeof(int));

  /* GPU */
  unsigned long int d_size =size*sizeof(int);
  int *d_num = NULL;
  err = cudaMalloc((void **)&d_num, d_size);
  if (err != cudaSuccess)  {
        fprintf(stderr, "Failed to allocate device memory d_space (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
   }


  /* Lunching Kernels */

  /* Gathering information from GPU */
   err= cudaGetDevice(&device);
   err= cudaGetDeviceProperties (&prop,device);
   
   int MaxThreads=prop.maxThreadsPerBlock;
   
  /* Blocks needed to generate the array of size elements if every block uses the maximum number of threads*/
   int nCudaBlocks=size/MaxThreads +1;
   printf("Launching %d of %d threads \n",nCudaBlocks,MaxThreads);
   fflush(NULL);

  /*Add the statement for lunching the kernel
   if it were a function in C, it would be written  int_generator(d_num,size); */


   

  /*Transfer memory back to host */
   gpuErrchk( cudaMemcpy(h_num,d_num,d_size, cudaMemcpyDeviceToHost));

   sprintf(fname,"integers.dat");
    write_output(fname,h_num,size);
 
  /* Free device global memory */
    err = cudaFree(d_num);
    if (err != cudaSuccess){
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

  /* Free host memory */
    free(h_num);

    /* Reset the device and exit
       cudaDeviceReset causes the driver to clean up all state. While
       not mandatory in normal operation, it is good practice.  It is also
       needed to ensure correct operation when the application is being
       profiled. Calling cudaDeviceReset causes all profile data to be
      flushed before the application exits
     */

    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Done\n");
return(0);
}
