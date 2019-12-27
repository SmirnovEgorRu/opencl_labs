#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void daxpy(const size_t n, const double a, __global const double *x, const size_t incx, __global double *y, const size_t incy) {
    int i = get_global_id(0);
    if (i * incy < n && i * incx < n)
         y[i * incy] = y[i * incy] + a * x[i * incx];
}