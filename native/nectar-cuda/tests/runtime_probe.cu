// A CUDA runtime program: finds the GPU, copies memory and launches a kernel. Run it on the
// real driver, through the tracer, and through Nectar to see what the CUDA runtime needs.
//
// Build: nvcc -arch=sm_120 -o runtime-probe.exe runtime_probe.cu

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void add(const float *a, const float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

#define CHECK(call)                                                                                \
    do                                                                                             \
    {                                                                                              \
        cudaError_t error = (call);                                                                \
        if (error != cudaSuccess)                                                                  \
        {                                                                                          \
            printf("FAIL %s: %s (%d)\n", #call, cudaGetErrorString(error), (int)error);            \
            return 1;                                                                              \
        }                                                                                          \
    } while (0)

int main()
{
    int version = 0;
    CHECK(cudaDriverGetVersion(&version));
    printf("driver     CUDA %d.%d\n", version / 1000, version % 1000 / 10);

    int count = 0;
    CHECK(cudaGetDeviceCount(&count));
    cudaDeviceProp properties;
    CHECK(cudaGetDeviceProperties(&properties, 0));
    printf("device 0   %s, compute %d.%d, %d devices\n", properties.name, properties.major, properties.minor, count);

    const int n = 1 << 20;
    const size_t bytes = n * sizeof(float);
    float *a = (float *)malloc(bytes), *b = (float *)malloc(bytes), *c = (float *)malloc(bytes);
    for (int i = 0; i < n; i++)
    {
        a[i] = (float)i;
        b[i] = 2.0f * i;
    }

    float *da, *db, *dc;
    CHECK(cudaMalloc(&da, bytes));
    CHECK(cudaMalloc(&db, bytes));
    CHECK(cudaMalloc(&dc, bytes));
    CHECK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice));
    printf("memory     copied 2 x %zu MiB to the GPU\n", bytes >> 20);

    add<<<(n + 255) / 256, 256>>>(da, db, dc, n);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++)
    {
        if (c[i] != 3.0f * i)
        {
            printf("FAIL c[%d] = %f, expected %f\n", i, c[i], 3.0f * i);
            return 1;
        }
    }
    printf("PASS kernel result matches\n");

    CHECK(cudaFree(da));
    CHECK(cudaFree(db));
    CHECK(cudaFree(dc));
    return 0;
}
