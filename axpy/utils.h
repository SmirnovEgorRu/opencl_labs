
#ifndef _OCL_LABS_UTILS_
#define _OCL_LABS_UTILS_


#include <random>
#include <algorithm>
#include <omp.h>


#define __CL_ENABLE_EXCEPTIONS
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.cpp>
#else
#include <CL/cl.hpp>
#endif

template<typename T>
void rand_f(T* ptr, size_t n, T start, T end, int seed)
{
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<T> distribution(start, end);
    std::generate(ptr, ptr + n, [&]()
    {
        return distribution(generator);
    });
}

template<typename FuncToPrepare, typename FuncToMeasure>
void measure(FuncToPrepare func1, FuncToMeasure func2, std::string str, size_t nIter = 10) {
    std::vector<double> v(nIter);
    double t1, t2;

    for(size_t i = 0; i < nIter; i++)
    {
        func1();
        t1 = omp_get_wtime();
        func2();
        t2 = omp_get_wtime();
        v[i] = (t2-t1)*1000;
    }

    double avg = 0;
    for(size_t i = 0; i < nIter; i++)
    {
        avg += v[i];
    }
    std::sort(v.begin(), v.end());
    printf("%-30s: %8.3f msec\n", str.c_str(), v[nIter/2]);
}

template<typename FuncToPrepare, typename FuncToMeasure>
void measure_2(FuncToPrepare func1, FuncToMeasure func2, std::string str, size_t nIter = 10) {
    std::vector<double> v(nIter);
    std::vector<double> v_2(nIter);
    double t1, t2;

    for(size_t i = 0; i < nIter; i++)
    {
        func1();
        t1 = omp_get_wtime();
        double t_kernel = func2();
        t2 = omp_get_wtime();
        v[i] = (t2-t1)*1000;
        v_2[i] = t_kernel;
    }

    std::sort(v.begin(), v.end());
    std::sort(v_2.begin(), v_2.end());
    printf("%-30s: %8.3f msec |  %8.3f msec\n", str.c_str(), v_2[nIter/2], v[nIter/2]);
}

#define RET_CODE_CHECK(retCode, func)                                      \
    retCode = func;                                                        \
    if (retCode) printf("Error: retCode = %d\n", static_cast<int>(retCode));

#define RET_CODE_FUNC_CHECK(retCode, func)                                 \
    func;                                                                  \
    if (retCode) printf("Error: retCode = %d\n", static_cast<int>(retCode));

#define RET_CODE_RETURN_CHECK(retCode, func, result)                       \
    result = func;                                                         \
    if (retCode) printf("Error: retCode = %d\n", static_cast<int>(retCode));

template <typename FPType>
std::string readKernel(std::string file) {
    std::ifstream ifs(file);
    std::string content{ std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>() };

    return content;
}

template <typename FPType>
void initializeKernel(cl_kernel& kernel, cl_context& context, cl_command_queue& queue,
    cl_device_id& dev_id, cl_program& program, cl_int& retCode, std::string kernel_file, cl_device_type deviceType) {
    cl_uint platformsCount = 0;
    clGetPlatformIDs(0, nullptr, &platformsCount);

    cl_platform_id* platforms = new cl_platform_id[platformsCount];
    clGetPlatformIDs(platformsCount, platforms, nullptr);

    cl_platform_id platform = platforms[0];
    cl_context_properties properties[3] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };

    context = clCreateContextFromType((platform == nullptr) ? nullptr : properties,
        deviceType, 0, 0, &retCode);
    if (retCode) {
        cl_platform_id platform = platforms[1];
        cl_context_properties properties[3] = {
            CL_CONTEXT_PLATFORM,
            (cl_context_properties)platform,
            0
        };
        RET_CODE_RETURN_CHECK(retCode, clCreateContextFromType((platform == nullptr) ? nullptr : properties,
            deviceType, 0, 0, &retCode), context)
    }

    size_t devices_count = 0;
    clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, 0, &devices_count);

    cl_device_id* devices = new cl_device_id[devices_count];
    clGetContextInfo(context, CL_CONTEXT_DEVICES, devices_count, devices, 0);
    dev_id = devices[0];

    char devName[128];
    clGetDeviceInfo(dev_id, CL_DEVICE_NAME, 128, devName, nullptr);

    RET_CODE_RETURN_CHECK(retCode, clCreateCommandQueueWithProperties(context, dev_id, 0, &retCode), queue)

    std::string content = readKernel<FPType>(kernel_file);
    size_t kernelLen = content.length();
    char* kernelSource = new char[kernelLen + 1];
    for (size_t i = 0; i < kernelLen; ++i)
        kernelSource[i] = content[i];
    kernelSource[kernelLen] = '\0';

    RET_CODE_RETURN_CHECK(retCode, clCreateProgramWithSource(context, 1, (const char**)&kernelSource,
        &kernelLen, &retCode), program)
    clBuildProgram(program, 1, &dev_id, 0, 0, 0);

    kernel = clCreateKernel(program, sizeof(FPType) == sizeof(double) ? "daxpy" : "saxpy", 0);

    delete[] kernelSource;
}

template<typename T>
class MemObj {
public:
    MemObj(size_t size, size_t align = 128): size_(size) {
        ptr_ = (T*)_mm_malloc(size*sizeof(T), align);
    }

    void init(int seed, double a = -1.0, double b = 1.0) {
        rand_f<T>(ptr_, size_, a, b, seed);
    }

    T* get() {
        return ptr_;
    }

    const T* get() const {
        return ptr_;
    }

    ~MemObj() {
        _mm_free(ptr_);
    }

    void copy(const T* src) {
        for(size_t i = 0; i < size_; i++) {
            ptr_[i] = src[i];
        }
    }

protected:
    T* ptr_;
    size_t size_;
};


#endif // _OCL_LABS_UTILS_