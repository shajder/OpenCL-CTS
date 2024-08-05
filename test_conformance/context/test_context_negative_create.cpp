//
// Copyright (c) 2024 The Khronos Group Inc.
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

#include "harness/testHarness.h"
#include "harness/errorHelpers.h"
#include "harness/typeWrappers.h"
#include <chrono>
#include <system_error>
#include <thread>
#include <vector>
#include <algorithm>

int test_context_negative_create(cl_device_id device, cl_context context,
                                 cl_command_queue queue, int num_elements)
{
    cl_int err = CL_SUCCESS;

    cl_uint num_platforms = 0;
    err = clGetPlatformIDs(16, nullptr, &num_platforms);
    test_error(err, "clGetPlatformIDs failed");


    num_platforms = std::min(num_platforms, (cl_uint)2);
    std::vector<cl_platform_id> platforms(num_platforms);

    err = clGetPlatformIDs(num_platforms, platforms.data(), &num_platforms);
    test_error(err, "clGetPlatformIDs failed");

    std::vector<cl_device_id> platform_devices;

    cl_uint num_devices = 0;
    for (int p = 0; p < (int)num_platforms; p++)
    {
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, nullptr,
                             &num_devices);
        test_error(err, "clGetDeviceIDs failed");

        std::vector<cl_device_id> devices(num_devices);
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, num_devices,
                             devices.data(), nullptr);
        test_error(err, "clGetDeviceIDs failed");

        platform_devices.push_back(devices.front());
    }

    if (platform_devices.size() < 2)
    {
        log_info("Can't find needed resources. Skipping the test.\n");
        return TEST_SKIPPED_ITSELF;
    }

    // Create secondary context
    clContextWrapper multi_dev_context =
        clCreateContext(0, platform_devices.size(), platform_devices.data(),
                        nullptr, nullptr, &err);
    test_error(err, "Failed to create context");

    test_failure_error(err, CL_INVALID_PROPERTY,
                       "Unexpected clCreateContext return");

    return TEST_PASS;
}
