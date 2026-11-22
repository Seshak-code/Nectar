// The probe's test kernel. add.ptx is generated from it:
//   nvcc -ptx -arch=compute_75 -o add.ptx add.cu
// PTX for compute 7.5 is compiled for whatever GPU runs it.

extern "C" __global__ void add(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}
