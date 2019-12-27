__kernel void saxpy(const size_t n, const float a, __global const float *x, const size_t incx, __global float *y, const size_t incy) {
    int i = get_global_id(0);
    if (i < n)
         y[i * incy] = y[i * incy] + a * x[i * incx];
}