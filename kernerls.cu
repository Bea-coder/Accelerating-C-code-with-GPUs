#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>


/**
 * CUDA Kernel Device code
 *
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
__global__ void int_generator (int *d_out,unsigned long int size){

    int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    int index_y = blockIdx.y * blockDim.y + threadIdx.y;
    int grid_width = gridDim.x * blockDim.x;

    /* map the two 2D indices to a single linear, 1D index */

    int gId  = index_y * grid_width + index_x;
    if(gId<size){
      d_out[gId]=gId;
    }     

}
