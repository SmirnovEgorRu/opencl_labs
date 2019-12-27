#include <CL/cl.h>
#include <iostream>
#include "cstring"

#define RET_CODE_CHECK(retCode, func)                                      \
    retCode = func;                                                        \
    if (retCode) printf("Error: retCode = %d\n", static_cast<int>(retCode));

#define RET_CODE_FUNC_CHECK(retCode, func)                                 \
    func;                                                                  \
    if (retCode) printf("Error: retCode = %d\n", static_cast<int>(retCode));

#define RET_CODE_RETURN_CHECK(retCode, func, result)                       \
    result = func;                                                         \
    if (retCode) printf("Error: retCode = %d\n", static_cast<int>(retCode));

const char* kernelSource =
"__kernel void addGlobalIndex(__global int *array, const size_t size) {\n"\
"    int globalID = get_global_id(0);                          "\
"    int localID  = get_local_id(0);                           "\
"    int groupID  = get_group_id(0);                           "\
"                                                              "\
"    if (globalID < size)                                      "\
"        array[globalID] += globalID;                          "\
"}";

int main() {
    cl_uint platformsCount = 0;
    clGetPlatformIDs(0, nullptr, &platformsCount);

    cl_platform_id* platforms = new cl_platform_id[platformsCount];
    clGetPlatformIDs(platformsCount, platforms, nullptr);

    for (cl_uint i = 0; i < platformsCount; ++i) {
        char platformsName[128];
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 128, platformsName, nullptr);
        std::cout << "Platforms " << i << ": " << platformsName << std::endl;

        cl_uint cpuCount = 0;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 0, nullptr, &cpuCount);
        cl_device_id* cpus = new cl_device_id[cpuCount];
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, cpuCount, cpus, nullptr);

        for (cl_uint j = 0; j < cpuCount; ++j) {
            char cpuName[128];
            clGetDeviceInfo(cpus[j], CL_DEVICE_NAME, 128, cpuName, nullptr);
            std::cout << "CPU: " << cpuName << std::endl;
        }

        cl_uint gpuCount = 0;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &gpuCount);
        cl_device_id* gpus = new cl_device_id[gpuCount];
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, gpuCount, gpus, nullptr);

        for (cl_uint j = 0; j < gpuCount; ++j) {
            char gpuName[128];
            clGetDeviceInfo(gpus[j], CL_DEVICE_NAME, 128, gpuName, nullptr);
            std::cout << "GPU: " << gpuName << std::endl;
        }
    }

    cl_int retCode = 0;

    cl_platform_id platform = platforms[0];
    cl_context_properties properties[3] = {
        CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform,
        0
    };
    cl_context context;
    RET_CODE_RETURN_CHECK(retCode, clCreateContextFromType((platform == nullptr) ? nullptr : properties,
        CL_DEVICE_TYPE_GPU, 0, 0, &retCode), context)

    size_t gpuCount = 0;
    RET_CODE_FUNC_CHECK(retCode, clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, 0, &gpuCount))

    cl_device_id* gpus = new cl_device_id[gpuCount];
    RET_CODE_FUNC_CHECK(retCode, clGetContextInfo(context, CL_CONTEXT_DEVICES, gpuCount, gpus, 0))
    cl_device_id gpu = gpus[0];

    cl_command_queue queue;
    RET_CODE_RETURN_CHECK(retCode, clCreateCommandQueueWithProperties(context, gpu, 0, &retCode), queue)

    size_t kernelLen = strlen(kernelSource);

    cl_program program;
    RET_CODE_RETURN_CHECK(retCode, clCreateProgramWithSource(context, 1, &kernelSource, &kernelLen, &retCode), program)
    clBuildProgram(program, 1, &gpu, 0, 0, 0);

    cl_kernel kernel = clCreateKernel(program, "addGlobalIndex", 0);

    const size_t size = 1024;
    int* array = new int[size];
    for (size_t i = 0; i < size; ++i)
        array[i] = 0;

    printf("array before:");
    for (size_t i = 0; i < size; ++i)
        printf(" %d", array[i]);
    printf("\n");

    size_t biteSize = sizeof(int) * size;

    cl_mem buffer;
    RET_CODE_RETURN_CHECK(retCode, clCreateBuffer(context, CL_MEM_READ_WRITE, biteSize, 0, &retCode), buffer)

    RET_CODE_CHECK(retCode, clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, biteSize, array, 0, 0, 0))

    RET_CODE_CHECK(retCode, clSetKernelArg(kernel, 0, sizeof(cl_mem), &buffer))

    RET_CODE_CHECK(retCode, clSetKernelArg(kernel, 1, sizeof(size_t), &size))

    size_t group;
    RET_CODE_CHECK(retCode, clGetKernelWorkGroupInfo(kernel, gpu, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &group, 0))

    printf("group = %zu, size = %zu\n", group, size);
    RET_CODE_CHECK(retCode, clEnqueueNDRangeKernel(queue, kernel, 1, 0, &size, &group, 0, 0, 0))

    RET_CODE_CHECK(retCode, clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, biteSize, array, 0, 0, 0))

    printf("array:");
    for (size_t i = 0; i < size; ++i)
        printf(" %d", array[i]);
    printf("\n");

    clReleaseMemObject(buffer);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}