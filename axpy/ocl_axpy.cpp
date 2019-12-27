#include <iostream>
#include <fstream>
#include <string>
#include <memory>
#include <stdlib.h>
#include <emmintrin.h>
#include "utils.h"


template<typename FPType>
void axpy(int n, FPType a, const FPType* const x, int incx, FPType* y, const int incy) {
    for(size_t i = 0; i < n; ++i) {
        y[i*incy] += a * x[i*incx];
    }
}

template<typename FPType>
void axpy_omp(int n, FPType a, const FPType* const x, int incx, FPType* y, const int incy) {
    #pragma omp parallel for
    for(size_t i = 0; i < n; ++i) {
        y[i*incy] += a * x[i*incx];
    }
}

template <typename FPType>
void setKernelArguments(const size_t n, const FPType a, const FPType* x, const size_t incx, FPType* y,
    const size_t incy, cl_kernel& kernel, cl_context& context, cl_command_queue& queue,
    cl_device_id& dev_id, cl_int& retCode, cl_mem& xBuffer, cl_mem& yBuffer, size_t& groupSize) {
    RET_CODE_CHECK(retCode, clGetKernelWorkGroupInfo(kernel, dev_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &groupSize, 0))

    size_t biteSize = sizeof(FPType) * (n / groupSize + !!(n % groupSize)) * groupSize;

    RET_CODE_CHECK(retCode, clSetKernelArg(kernel, 0, sizeof(size_t), &n))
    RET_CODE_CHECK(retCode, clSetKernelArg(kernel, 1, sizeof(FPType), &a))

    RET_CODE_RETURN_CHECK(retCode, clCreateBuffer(context, CL_MEM_READ_ONLY, biteSize, 0, &retCode), xBuffer)
    RET_CODE_CHECK(retCode, clEnqueueWriteBuffer(queue, xBuffer, CL_TRUE, 0, biteSize, x, 0, 0, 0))
    RET_CODE_CHECK(retCode, clSetKernelArg(kernel, 2, sizeof(cl_mem), &xBuffer))

    RET_CODE_CHECK(retCode, clSetKernelArg(kernel, 3, sizeof(size_t), &incx))

    RET_CODE_RETURN_CHECK(retCode, clCreateBuffer(context, CL_MEM_READ_WRITE, biteSize, 0, &retCode), yBuffer)
    RET_CODE_CHECK(retCode, clEnqueueWriteBuffer(queue, yBuffer, CL_TRUE, 0, biteSize, y, 0, 0, 0))
    RET_CODE_CHECK(retCode, clSetKernelArg(kernel, 4, sizeof(cl_mem), &yBuffer))

    RET_CODE_CHECK(retCode, clSetKernelArg(kernel, 5, sizeof(size_t), &incy))
}


template <typename FPType>
double opencl_axpy(const size_t n, const FPType a, const FPType* x, const size_t incx, FPType* y, const size_t incy,
              cl_device_type deviceType) {
    cl_context context;
    cl_command_queue queue;
    cl_kernel kernel;
    cl_device_id dev_id;
    cl_program program;
    cl_mem xBuffer, yBuffer;
    cl_int retCode = 0;
    size_t groupSize = 1024;

    std::string str(sizeof(FPType) == sizeof(double) ? "daxpy_kernel.cl" : "saxpy_kernel.cl");
    initializeKernel<FPType>(kernel, context, queue, dev_id, program, retCode, str, deviceType);
    setKernelArguments(n, a, x, incx, y, incy, kernel, context, queue, dev_id, retCode, xBuffer, yBuffer, groupSize);

    size_t nWorkItems = (n / groupSize + !!(n % groupSize)) * groupSize;
    cl_event event;
    auto t1 = omp_get_wtime();
    RET_CODE_CHECK(retCode, clEnqueueNDRangeKernel(queue, kernel, 1, 0, &nWorkItems, &groupSize, 0, 0, &event))
    clWaitForEvents(1, &event);
    auto t2 = omp_get_wtime();

    RET_CODE_CHECK(retCode, clEnqueueReadBuffer(queue, yBuffer, CL_TRUE, 0, sizeof(FPType) * n, y, 0, 0, 0))

    clReleaseMemObject(xBuffer);
    clReleaseMemObject(yBuffer);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    return (t2-t1)*1000;
}


template<typename FPType>
void check_axpy(const size_t n, const FPType a, const FPType* x, const size_t incx, FPType* y, const size_t incy, FPType* y_after ) {
    MemObj<FPType> Y(n);
    Y.copy(y);

    axpy(n, a, x, incx, Y.get(), incy);

    for(size_t i = 0; i < n; ++i) {
        if (std::abs(y_after[i] - Y.get()[i]) > 0.001) {
            printf("Error!\n");
            break;
        }
    }
}

template<typename FPType>
void run(size_t n_elems) {
    MemObj<FPType> Y(n_elems);
    MemObj<FPType> Y_copy(n_elems);
    MemObj<FPType> X(n_elems);
    FPType a = 2.0;
    int incx = 1;
    int incy = 1;

    Y.init(666);
    X.init(666);
    Y_copy.copy(Y.get());

    measure(
        [&]() { Y.copy(Y_copy.get()); },
        [&]() { axpy(n_elems, a, X.get(), incx, Y.get(), incy); },
        "Single-thread axpy", 50
    );
    check_axpy(n_elems, a, X.get(), incx, Y_copy.get(), incy, Y.get());

    measure(
        [&]() { Y.copy(Y_copy.get()); },
        [&]() { axpy_omp(n_elems, a, X.get(), incx, Y.get(), incy); },
        "Multi-thread axpy", 50
    );
    check_axpy(n_elems, a, X.get(), incx, Y_copy.get(), incy, Y.get());

    measure_2(
        [&]() { Y.copy(Y_copy.get()); },
        [&]() { return opencl_axpy(n_elems, a, X.get(), 1, Y.get(), 1, CL_DEVICE_TYPE_CPU); },
        "OpenCL CPU axpy", 10
    );
    check_axpy(n_elems, a, X.get(), incx, Y_copy.get(), incy, Y.get());

    measure_2(
        [&]() { Y.copy(Y_copy.get()); },
        [&]() { return opencl_axpy(n_elems, a, X.get(), 1, Y.get(), 1, CL_DEVICE_TYPE_GPU); },
        "OpenCL GPU axpy", 10
    );
    check_axpy(n_elems, a, X.get(), incx, Y_copy.get(), incy, Y.get());
}

int main( int argc, char** argv ) {
    constexpr size_t N_ELEMENTS = 5*1024*1024;

    printf("FLOAT:\n");
    run<float> (N_ELEMENTS);
    printf("==================================================\n");
    printf("DOUBLE:\n");
    run<double> (N_ELEMENTS);
}