//
// Copyright (c) 2022 The Khronos Group Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
#include "basic_command_buffer.h"

#include <algorithm>
#include <cstring>
#include <vector>


//--------------------------------------------------------------------------
BasicCommandBufferTest::BasicCommandBufferTest(cl_device_id device,
                                               cl_context context,
                                               cl_command_queue queue)
    : CommandBufferTestBase(device), context(context), queue(nullptr),
      num_elements(0), simultaneous_use_support(false),
      out_of_order_support(false), queue_out_of_order_support(false),
      // try to use simultaneous path by default
      simultaneous_use_requested(true),
      // due to simultaneous cases extend buffer size
      buffer_size_multiplier(1), command_buffer(this)
{
    cl_int error = clRetainCommandQueue(queue);
    if (error != CL_SUCCESS)
    {
        throw std::runtime_error("clRetainCommandQueue failed\n");
    }
    this->queue = queue;
}

//--------------------------------------------------------------------------
bool BasicCommandBufferTest::Skip()
{
    cl_command_queue_properties required_properties;
    cl_int error = clGetDeviceInfo(
        device, CL_DEVICE_COMMAND_BUFFER_REQUIRED_QUEUE_PROPERTIES_KHR,
        sizeof(required_properties), &required_properties, NULL);
    test_error(error,
               "Unable to query "
               "CL_DEVICE_COMMAND_BUFFER_REQUIRED_QUEUE_PROPERTIES_KHR");

    cl_command_queue_properties supported_properties;
    error = clGetDeviceInfo(
        device, CL_DEVICE_COMMAND_BUFFER_SUPPORTED_QUEUE_PROPERTIES_KHR,
        sizeof(supported_properties), &supported_properties, NULL);
    test_error(error,
               "Unable to query "
               "CL_DEVICE_COMMAND_BUFFER_SUPPORTED_QUEUE_PROPERTIES_KHR");

    cl_command_queue_properties queue_properties;
    error = clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES,
                            sizeof(queue_properties), &queue_properties, NULL);
    test_error(error, "Unable to query CL_DEVICE_QUEUE_PROPERTIES");
    queue_out_of_order_support =
        queue_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;

    // Query if device supports simultaneous use
    cl_device_command_buffer_capabilities_khr capabilities;
    error = clGetDeviceInfo(device, CL_DEVICE_COMMAND_BUFFER_CAPABILITIES_KHR,
                            sizeof(capabilities), &capabilities, NULL);
    test_error(error,
               "Unable to query CL_DEVICE_COMMAND_BUFFER_CAPABILITIES_KHR");
    simultaneous_use_support = simultaneous_use_requested
        && (capabilities & CL_COMMAND_BUFFER_CAPABILITY_SIMULTANEOUS_USE_KHR)
            != 0;
    out_of_order_support =
        supported_properties & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
    device_side_enqueue_support =
        (capabilities & CL_COMMAND_BUFFER_CAPABILITY_DEVICE_SIDE_ENQUEUE_KHR)
        != 0;

    // Skip if queue properties don't contain those required
    return required_properties != (required_properties & queue_properties);
}

//--------------------------------------------------------------------------
cl_int BasicCommandBufferTest::SetUpKernel()
{
    cl_int error = CL_SUCCESS;

    // Kernel performs a parallel copy from an input buffer to output buffer
    // is created.
    const char *kernel_str =
        R"(
  __kernel void copy(__global int* in, __global int* out, __global int* offset) {
      size_t id = get_global_id(0);
      int ind = offset[0] + id;
      out[ind] = in[ind];
  })";

    error = create_single_kernel_helper_create_program(context, &program, 1,
                                                       &kernel_str);
    test_error(error, "Failed to create program with source");

    error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    test_error(error, "Failed to build program");

    kernel = clCreateKernel(program, "copy", &error);
    test_error(error, "Failed to create copy kernel");

    return CL_SUCCESS;
}

//--------------------------------------------------------------------------
cl_int BasicCommandBufferTest::SetUpKernelArgs()
{
    cl_int error = CL_SUCCESS;
    in_mem =
        clCreateBuffer(context, CL_MEM_READ_ONLY,
                       sizeof(cl_int) * num_elements * buffer_size_multiplier,
                       nullptr, &error);
    test_error(error, "clCreateBuffer failed");

    out_mem =
        clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                       sizeof(cl_int) * num_elements * buffer_size_multiplier,
                       nullptr, &error);
    test_error(error, "clCreateBuffer failed");

    cl_int offset = 0;
    off_mem = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                             sizeof(cl_int), &offset, &error);
    test_error(error, "clCreateBuffer failed");

    error = clSetKernelArg(kernel, 0, sizeof(in_mem), &in_mem);
    test_error(error, "clSetKernelArg failed");

    error = clSetKernelArg(kernel, 1, sizeof(out_mem), &out_mem);
    test_error(error, "clSetKernelArg failed");

    error = clSetKernelArg(kernel, 2, sizeof(off_mem), &off_mem);
    test_error(error, "clSetKernelArg failed");

    return CL_SUCCESS;
}

//--------------------------------------------------------------------------
cl_int BasicCommandBufferTest::SetUp(int elements)
{
    cl_int error = init_extension_functions();
    if (error != CL_SUCCESS)
    {
        return error;
    }

    if (elements <= 0)
    {
        return CL_INVALID_VALUE;
    }
    num_elements = static_cast<size_t>(elements);

    error = SetUpKernel();
    test_error(error, "SetUpKernel failed");

    error = SetUpKernelArgs();
    test_error(error, "SetUpKernelArgs failed");

    if (simultaneous_use_support)
    {
        cl_command_buffer_properties_khr properties[3] = {
            CL_COMMAND_BUFFER_FLAGS_KHR, CL_COMMAND_BUFFER_SIMULTANEOUS_USE_KHR,
            0
        };
        command_buffer =
            clCreateCommandBufferKHR(1, &queue, properties, &error);
    }
    else
    {
        command_buffer = clCreateCommandBufferKHR(1, &queue, nullptr, &error);
    }
    test_error(error, "clCreateCommandBufferKHR failed");

    return CL_SUCCESS;
}

cl_int MultiFlagCreationTest::Run()
{
    cl_command_buffer_properties_khr flags = 0;
    cl_int error = CL_SUCCESS;

    // First try to find multiple flags that are supported by the driver and
    // device.
    if (simultaneous_use_support)
    {
        flags |= CL_COMMAND_BUFFER_SIMULTANEOUS_USE_KHR;
    }

    if (is_extension_available(
            device, CL_KHR_COMMAND_BUFFER_MULTI_DEVICE_EXTENSION_NAME))
    {
        flags |= CL_COMMAND_BUFFER_DEVICE_SIDE_SYNC_KHR;
    }

    if (is_extension_available(
            device, CL_KHR_COMMAND_BUFFER_MUTABLE_DISPATCH_EXTENSION_NAME))
    {
        flags |= CL_COMMAND_BUFFER_MUTABLE_KHR;
    }

    cl_command_buffer_properties_khr props[] = { CL_COMMAND_BUFFER_FLAGS_KHR,
                                                 flags, 0 };

    command_buffer = clCreateCommandBufferKHR(1, &queue, props, &error);
    test_error(error, "clCreateCommandBufferKHR failed");

    return CL_SUCCESS;
};

cl_int BasicEnqueueTest::Run()
{

    cl_int error = clCommandNDRangeKernelKHR(
        command_buffer, nullptr, nullptr, kernel, 1, nullptr, &num_elements,
        nullptr, 0, nullptr, nullptr, nullptr);
    test_error(error, "clCommandNDRangeKernelKHR failed");

    error = clFinalizeCommandBufferKHR(command_buffer);
    test_error(error, "clFinalizeCommandBufferKHR failed");

    const cl_int pattern = 42;
    error = clEnqueueFillBuffer(queue, in_mem, &pattern, sizeof(cl_int), 0,
                                data_size(), 0, nullptr, nullptr);
    test_error(error, "clEnqueueFillBuffer failed");

    error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0, nullptr,
                                      nullptr);
    test_error(error, "clEnqueueCommandBufferKHR failed");

    std::vector<cl_int> output_data_1(num_elements);
    error = clEnqueueReadBuffer(queue, out_mem, CL_TRUE, 0, data_size(),
                                output_data_1.data(), 0, nullptr, nullptr);
    test_error(error, "clEnqueueReadBuffer failed");

    for (size_t i = 0; i < num_elements; i++)
    {
        CHECK_VERIFICATION_ERROR(pattern, output_data_1[i], i);
    }

    const cl_int new_pattern = 12;
    error = clEnqueueFillBuffer(queue, in_mem, &new_pattern, sizeof(cl_int), 0,
                                data_size(), 0, nullptr, nullptr);
    test_error(error, "clEnqueueFillBuffer failed");

    error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0, nullptr,
                                      nullptr);
    test_error(error, "clEnqueueCommandBufferKHR failed");

    std::vector<cl_int> output_data_2(num_elements);
    error = clEnqueueReadBuffer(queue, out_mem, CL_TRUE, 0, data_size(),
                                output_data_2.data(), 0, nullptr, nullptr);
    test_error(error, "clEnqueueReadBuffer failed");

    for (size_t i = 0; i < num_elements; i++)
    {
        CHECK_VERIFICATION_ERROR(new_pattern, output_data_2[i], i);
    }

    return CL_SUCCESS;
};

cl_int MixedCommandsTest::Run()
{
    cl_int error;
    const size_t iterations = 4;
    clMemWrapper result_mem =
        clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_int) * iterations,
                       nullptr, &error);
    test_error(error, "clCreateBuffer failed");

    const cl_int pattern_base = 42;
    for (size_t i = 0; i < iterations; i++)
    {
        const cl_int pattern = pattern_base + i;
        cl_int error = clCommandFillBufferKHR(
            command_buffer, nullptr, nullptr, in_mem, &pattern, sizeof(cl_int),
            0, data_size(), 0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandFillBufferKHR failed");

        error = clCommandNDRangeKernelKHR(
            command_buffer, nullptr, nullptr, kernel, 1, nullptr, &num_elements,
            nullptr, 0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandNDRangeKernelKHR failed");

        const size_t result_offset = i * sizeof(cl_int);
        error = clCommandCopyBufferKHR(
            command_buffer, nullptr, nullptr, out_mem, result_mem, 0,
            result_offset, sizeof(cl_int), 0, nullptr, nullptr, nullptr);
        test_error(error, "clCommandCopyBufferKHR failed");
    }

    error = clFinalizeCommandBufferKHR(command_buffer);
    test_error(error, "clFinalizeCommandBufferKHR failed");

    error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0, nullptr,
                                      nullptr);
    test_error(error, "clEnqueueCommandBufferKHR failed");

    std::vector<cl_int> result_data(num_elements);
    error = clEnqueueReadBuffer(queue, result_mem, CL_TRUE, 0,
                                iterations * sizeof(cl_int), result_data.data(),
                                0, nullptr, nullptr);
    test_error(error, "clEnqueueReadBuffer failed");

    for (size_t i = 0; i < iterations; i++)
    {
        const cl_int ref = pattern_base + i;
        CHECK_VERIFICATION_ERROR(ref, result_data[i], i);
    }

    return CL_SUCCESS;
}

cl_int ExplicitFlushTest::Run()
{
    cl_int error = clCommandNDRangeKernelKHR(
        command_buffer, nullptr, nullptr, kernel, 1, nullptr, &num_elements,
        nullptr, 0, nullptr, nullptr, nullptr);
    test_error(error, "clCommandNDRangeKernelKHR failed");

    error = clFinalizeCommandBufferKHR(command_buffer);
    test_error(error, "clFinalizeCommandBufferKHR failed");

    const cl_int pattern_A = 42;
    error = clEnqueueFillBuffer(queue, in_mem, &pattern_A, sizeof(cl_int), 0,
                                data_size(), 0, nullptr, nullptr);
    test_error(error, "clEnqueueFillBuffer failed");

    error = clFlush(queue);
    test_error(error, "clFlush failed");

    error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0, nullptr,
                                      nullptr);
    test_error(error, "clEnqueueCommandBufferKHR failed");

    std::vector<cl_int> output_data_A(num_elements);
    error = clEnqueueReadBuffer(queue, out_mem, CL_FALSE, 0, data_size(),
                                output_data_A.data(), 0, nullptr, nullptr);
    test_error(error, "clEnqueueReadBuffer failed");

    const cl_int pattern_B = 0xA;
    error = clEnqueueFillBuffer(queue, in_mem, &pattern_B, sizeof(cl_int), 0,
                                data_size(), 0, nullptr, nullptr);
    test_error(error, "clEnqueueFillBuffer failed");

    error = clFlush(queue);
    test_error(error, "clFlush failed");

    error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0, nullptr,
                                      nullptr);
    test_error(error, "clEnqueueCommandBufferKHR failed");

    error = clFlush(queue);
    test_error(error, "clFlush failed");

    std::vector<cl_int> output_data_B(num_elements);
    error = clEnqueueReadBuffer(queue, out_mem, CL_FALSE, 0, data_size(),
                                output_data_B.data(), 0, nullptr, nullptr);
    test_error(error, "clEnqueueReadBuffer failed");

    error = clFinish(queue);
    test_error(error, "clFinish failed");

    for (size_t i = 0; i < num_elements; i++)
    {
        CHECK_VERIFICATION_ERROR(pattern_A, output_data_A[i], i);

        CHECK_VERIFICATION_ERROR(pattern_B, output_data_B[i], i);
    }
    return CL_SUCCESS;
}

bool ExplicitFlushTest::Skip()
{
    return BasicCommandBufferTest::Skip() || !simultaneous_use_support;
}

cl_int InterleavedEnqueueTest::Run()
{
    cl_int error = clCommandNDRangeKernelKHR(
        command_buffer, nullptr, nullptr, kernel, 1, nullptr, &num_elements,
        nullptr, 0, nullptr, nullptr, nullptr);
    test_error(error, "clCommandNDRangeKernelKHR failed");

    error = clFinalizeCommandBufferKHR(command_buffer);
    test_error(error, "clFinalizeCommandBufferKHR failed");

    cl_int pattern = 42;
    error = clEnqueueFillBuffer(queue, in_mem, &pattern, sizeof(cl_int), 0,
                                data_size(), 0, nullptr, nullptr);
    test_error(error, "clEnqueueFillBuffer failed");

    error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0, nullptr,
                                      nullptr);
    test_error(error, "clEnqueueCommandBufferKHR failed");

    pattern = 0xABCD;
    error = clEnqueueFillBuffer(queue, in_mem, &pattern, sizeof(cl_int), 0,
                                data_size(), 0, nullptr, nullptr);
    test_error(error, "clEnqueueFillBuffer failed");

    error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0, nullptr,
                                      nullptr);
    test_error(error, "clEnqueueCommandBufferKHR failed");

    error = clEnqueueCopyBuffer(queue, in_mem, out_mem, 0, 0, data_size(), 0,
                                nullptr, nullptr);
    test_error(error, "clEnqueueCopyBuffer failed");

    std::vector<cl_int> output_data(num_elements);
    error = clEnqueueReadBuffer(queue, out_mem, CL_TRUE, 0, data_size(),
                                output_data.data(), 0, nullptr, nullptr);
    test_error(error, "clEnqueueReadBuffer failed");

    for (size_t i = 0; i < num_elements; i++)
    {
        CHECK_VERIFICATION_ERROR(pattern, output_data[i], i);
    }

    return CL_SUCCESS;
}

bool InterleavedEnqueueTest::Skip()
{
    return BasicCommandBufferTest::Skip() || !simultaneous_use_support;
}

cl_int EnqueueAndReleaseTest::Run()
{
    cl_int error = clCommandNDRangeKernelKHR(
        command_buffer, nullptr, nullptr, kernel, 1, nullptr, &num_elements,
        nullptr, 0, nullptr, nullptr, nullptr);
    test_error(error, "clCommandNDRangeKernelKHR failed");

    error = clFinalizeCommandBufferKHR(command_buffer);
    test_error(error, "clFinalizeCommandBufferKHR failed");

    cl_int pattern = 42;
    error = clEnqueueFillBuffer(queue, in_mem, &pattern, sizeof(cl_int), 0,
                                data_size(), 0, nullptr, nullptr);
    test_error(error, "clEnqueueFillBuffer failed");

    error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0, nullptr,
                                      nullptr);
    test_error(error, "clEnqueueCommandBufferKHR failed");

    // Calls release on cl_command_buffer_khr handle inside wrapper class, and
    // sets the handle to nullptr, so that release doesn't get called again at
    // end of test when wrapper object is destroyed.
    command_buffer.reset();

    std::vector<cl_int> output_data(num_elements);
    error = clEnqueueReadBuffer(queue, out_mem, CL_TRUE, 0, data_size(),
                                output_data.data(), 0, nullptr, nullptr);
    test_error(error, "clEnqueueReadBuffer failed");

    for (size_t i = 0; i < num_elements; i++)
    {
        CHECK_VERIFICATION_ERROR(pattern, output_data[i], i);
    }

    return CL_SUCCESS;
}
