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

#include <harness/os_helpers.h>

#include "basic_command_buffer.h"
#include "procs.h"

#if !defined(_WIN32)
#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif
#include <unistd.h>
#define streamDup(fd1) dup(fd1)
#define streamDup2(fd1, fd2) dup2(fd1, fd2)
#endif
#include <limits.h>
#include <time.h>

#if defined(_WIN32)
#include <io.h>
#define streamDup(fd1) _dup(fd1)
#define streamDup2(fd1, fd2) _dup2(fd1, fd2)
#endif

#include <vector>
#include <fstream>
#include <stdio.h>

namespace {

////////////////////////////////////////////////////////////////////////////////
// printf tests for cl_khr_command_buffer which handles below cases:
// -test cases for device side Printf
// -test cases for device side printf with a simultaneous use command-buffer

template < bool simul_use >
struct CommandBufferPrintfTest : public BasicCommandBufferTest
{
    CommandBufferPrintfTest(cl_device_id device, cl_context context,
                            cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue),
          trigger_event(nullptr), file_descriptor(0)
    {
        simultaneous_use_requested = simul_use;
        if (simul_use)
        {
            buffer_size_multiplier = num_test_iters;
        }
    }

    //--------------------------------------------------------------------------
    void ReleaseOutputStream(int fd)
    {
        fflush(stdout);
        streamDup2(fd, fileno(stdout));
        close(fd);
    }

    //--------------------------------------------------------------------------
    int AcquireOutputStream(int* error)
    {
        int fd = streamDup(fileno(stdout));
        *error = 0;
        if (!freopen(temp_filename.c_str(), "wt", stdout))
        {
            ReleaseOutputStream(fd);
            *error = -1;
        }
        return fd;
    }

    //--------------------------------------------------------------------------
    void GetAnalysisBuffer(std::stringstream& buffer)
    {
        std::ifstream fp(temp_filename, std::ios::in);
        if (fp.is_open())
        {
            buffer << fp.rdbuf();
        }
    }

    //--------------------------------------------------------------------------
    void PurgeTempFile()
    {
        std::ofstream ofs(temp_filename,
                          std::ofstream::out | std::ofstream::trunc);
        ofs.close();
    }

    //--------------------------------------------------------------------------
    bool Skip() override
    {
        return (simultaneous_use_requested && !simultaneous_use_support)
            || BasicCommandBufferTest::Skip();
    }

    //--------------------------------------------------------------------------
    cl_int SetUpKernel() override
    {
        cl_int error = CL_SUCCESS;

        const char* kernel_str =
            R"(
      __kernel void print(__global char* in, __global char* out, __global int* offset)
      {
          size_t id = get_global_id(0);
          int ind = offset[0] + offset[1] * id;
          for(int i=0; i<offset[1]; i++) out[ind+i] = in[i];
          printf("%s", in);
      })";

        error = create_single_kernel_helper_create_program(context, &program, 1,
                                                           &kernel_str);
        test_error(error, "Failed to create program with source");

        error = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
        test_error(error, "Failed to build program");

        kernel = clCreateKernel(program, "print", &error);
        test_error(error, "Failed to create print kernel");

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    size_t data_size() const override
    {
        return sizeof(cl_char) * num_elements * buffer_size_multiplier
            * max_pattern_length;
    }

    //--------------------------------------------------------------------------
    cl_int SetUpKernelArgs() override
    {
        cl_int error = CL_SUCCESS;

        in_mem = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                sizeof(cl_char) * (max_pattern_length + 1),
                                nullptr, &error);
        test_error(error, "clCreateBuffer failed");

        out_mem = clCreateBuffer(context, CL_MEM_WRITE_ONLY, data_size(),
                                 nullptr, &error);
        test_error(error, "clCreateBuffer failed");

        cl_int offset[] = { 0, max_pattern_length };
        off_mem =
            clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                           sizeof(offset), offset, &error);
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
    cl_int SetUp(int elements) override
    {
        // Query if device supports simultaneous use
        cl_device_command_buffer_capabilities_khr capabilities;
        cl_int error =
            clGetDeviceInfo(device, CL_DEVICE_COMMAND_BUFFER_CAPABILITIES_KHR,
                            sizeof(capabilities), &capabilities, NULL);
        test_error(
            error,
            "Unable to query CL_COMMAND_BUFFER_CAPABILITY_KERNEL_PRINTF_KHR");

        if ((capabilities & CL_COMMAND_BUFFER_CAPABILITY_SIMULTANEOUS_USE_KHR)
            == 0)
        {
            log_error(
                "Device capability "
                "CL_COMMAND_BUFFER_CAPABILITY_KERNEL_PRINTF_KHR not supported");
            return CL_INVALID_DEVICE_TYPE;
        }

        temp_filename = get_temp_filename();
        if (temp_filename.empty())
        {
            log_error("get_temp_filename failed\n");
            return -1;
        }

        return BasicCommandBufferTest::SetUp(elements);
    }

    //--------------------------------------------------------------------------
    cl_int Run() override
    {
        cl_int error = CL_SUCCESS;

        // record command buffer with primary queue
        error = RecordCommandBuffer();
        test_error(error, "RecordCommandBuffer failed");

        if (simultaneous_use_support)
        {
          // enque simultaneous command-buffers with substitute queue
          error = RunSimultaneous();
          test_error(error, "RunSimultaneous failed");
        }
        else
        {
          // enque single command-buffer with substitute queue
          error = RunSingle();
          test_error(error, "RunSingle failed");
        }

        std::remove(temp_filename.c_str());

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RecordCommandBuffer()
    {
      cl_int error = CL_SUCCESS;

      error = clCommandNDRangeKernelKHR(
          command_buffer, nullptr, nullptr, kernel, 1, nullptr, &num_elements,
          nullptr, 0, nullptr, nullptr, nullptr);
      test_error(error, "clCommandNDRangeKernelKHR failed");

      error = clFinalizeCommandBufferKHR(command_buffer);
      test_error(error, "clFinalizeCommandBufferKHR failed");
      return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    int WaitForEvent(cl_event* event)
    {
        cl_int status = clWaitForEvents(1, event);
        if (status != CL_SUCCESS)
        {
            log_error("clWaitForEvents failed");
            return status;
        }

        status = clReleaseEvent(*event);
        if (status != CL_SUCCESS)
        {
            log_error("clReleaseEvent failed. (*event)");
            return status;
        }
        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int EnqueueSinglePass(const std::string& pattern,
                             std::vector<cl_char>& output_data)
    {
      cl_int error = CL_SUCCESS;
      auto in_mem_size = sizeof(cl_char) * (pattern.size() + 1);
      error = clEnqueueWriteBuffer(queue, in_mem, CL_TRUE, 0, in_mem_size,
                                   &pattern[0], 0, nullptr, nullptr);
      test_error(error, "clEnqueueFillBuffer failed");

      cl_int offset[] = { 0, pattern.size() };
      error = clEnqueueWriteBuffer(queue, off_mem, CL_TRUE, 0, sizeof(offset),
                                   offset, 0, nullptr, nullptr);
      test_error(error, "clEnqueueFillBuffer failed");

      file_descriptor = AcquireOutputStream(&error);
      if (error != 0)
      {
          log_error("Error while redirection stdout to file");
          return TEST_FAIL;
      }

      cl_event wait_event;
      error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0, nullptr,
                                        &wait_event);
      if (error != CL_SUCCESS)
      {
          ReleaseOutputStream(file_descriptor);
          log_error("clEnqueueCommandBufferKHR failed errcode: %d\n", error);
          return TEST_FAIL;
      }

      fflush(stdout);
      error = clFlush(queue);
      if (error != CL_SUCCESS)
      {
          ReleaseOutputStream(file_descriptor);
          log_error("clFlush failed\n");
          return TEST_FAIL;
      }

      // Wait until kernel finishes its execution and (thus) the output printed
      // from the kernel is immediately printed
      error = WaitForEvent(&wait_event);
      if (error != CL_SUCCESS)
      {
          ReleaseOutputStream(file_descriptor);
          log_error("\n WaitForEvent failed errcode:%d\n", error);
          return TEST_FAIL;
      }

      ReleaseOutputStream(file_descriptor);

      error = clEnqueueReadBuffer(queue, out_mem, CL_FALSE, 0, data_size(),
                                  output_data.data(), 0, nullptr, nullptr);
      test_error(error, "clEnqueueReadBuffer failed");

      error = clFinish(queue);
      test_error(error, "clFinish failed");

      std::stringstream sstr;
      GetAnalysisBuffer(sstr);
      if (sstr.str().size() != num_elements * pattern.size())
      {
          log_error("GetAnalysisBuffer failed\n");
          return TEST_FAIL;
      }

      for (size_t i = 0; i < num_elements * pattern.size(); i++)
      {
          CHECK_VERIFICATION_ERROR(sstr.str().at(i), output_data[i], i);
      }

      return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RunSingle()
    {
        cl_int error = CL_SUCCESS;
        std::vector<cl_char> output_data(num_elements * max_pattern_length);

        for (unsigned i = 0; i < num_test_iters; i++)
        {
            unsigned pattern_length =
                std::max(min_pattern_length, rand() % max_pattern_length);
            char pattern_character = 'a' + rand() % 26;
            std::string pattern(pattern_length, pattern_character);
            error = EnqueueSinglePass(pattern, output_data);
            test_error(error, "EnqueueSinglePass failed");

            output_data.assign(output_data.size(), 0);
            PurgeTempFile();
        }

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    struct SimulPassData
    {
        std::string pattern;
        cl_int offset;
        std::vector<cl_char> output_buffer;
    };

    //--------------------------------------------------------------------------
    cl_int EnqueueSimultaneousPass(SimulPassData& pd)
    {
        // the same patter for both
        auto in_mem_size = sizeof(cl_char) * (pd.pattern.size() + 1);
        cl_int error =
            clEnqueueWriteBuffer(queue, in_mem, CL_FALSE, 0, in_mem_size,
                                 &pd.pattern[0], 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");


        cl_int offset[] = { pd.offset, pd.pattern.size() };
        error =
            clEnqueueWriteBuffer(queue, off_mem, CL_FALSE, 0, sizeof(offset),
                                 offset, 0, nullptr, nullptr);
        test_error(error, "clEnqueueFillBuffer failed");

        if (!trigger_event)
        {
            trigger_event = clCreateUserEvent(context, &error);
            test_error(error, "clCreateUserEvent failed");
        }

        error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 1,
                                          &trigger_event, nullptr);
        test_error(error, "clEnqueueCommandBufferKHR failed");

        error = clEnqueueReadBuffer(
            queue, out_mem, CL_FALSE, pd.offset * sizeof(cl_char),
            pd.output_buffer.size() * sizeof(cl_char), pd.output_buffer.data(),
            0, nullptr, nullptr);

        test_error(error, "clEnqueueReadBuffer failed");

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RunSimultaneous()
    {
        cl_int error = CL_SUCCESS;
        cl_int offset = static_cast<cl_int>(num_elements * max_pattern_length);

        std::vector<SimulPassData> simul_passes(num_test_iters);

        // the same pattern character for all simultaneous command buffers
        char pattern_character = 'a' + rand() % 26;

        cl_int total_pattern_coverage = 0;
        for (unsigned i = 0; i < num_test_iters; i++)
        {
            unsigned pattern_length =
                std::max(min_pattern_length, rand() % max_pattern_length);
            std::string pattern(pattern_length, pattern_character);
            simul_passes[i] = { pattern, i * offset,
                                std::vector<cl_char>(num_elements
                                                     * pattern_length) };
            total_pattern_coverage += simul_passes[i].output_buffer.size();
        };

        // enqueue read/write and command buffer operations
        for (auto&& pass : simul_passes)
        {
            error = EnqueueSimultaneousPass(pass);
            test_error(error, "EnqueuePass failed");
        }

        // takeover stdout stream
        file_descriptor = AcquireOutputStream(&error);
        if (error != 0)
        {
            log_error("Error while redirection stdout to file");
            return TEST_FAIL;
        }

        // execute command buffers
        error = clSetUserEventStatus(trigger_event, CL_COMPLETE);
        if (error != CL_SUCCESS)
        {
            ReleaseOutputStream(file_descriptor);
            log_error("clSetUserEventStatus failed\n");
            return TEST_FAIL;
        }

        // flush streams
        fflush(stdout);
        error = clFlush(queue);
        if (error != CL_SUCCESS)
        {
            ReleaseOutputStream(file_descriptor);
            log_error("clFlush failed\n");
            return TEST_FAIL;
        }

        // finish command queue
        error = clFinish(queue);
        if (error != CL_SUCCESS)
        {
            ReleaseOutputStream(file_descriptor);
            log_error("clFinish failed\n");
            return TEST_FAIL;
        }

        ReleaseOutputStream(file_descriptor);

        std::stringstream sstr;
        GetAnalysisBuffer(sstr);
        if (sstr.str().size() != total_pattern_coverage)
        {
            log_error("GetAnalysisBuffer failed\n");
            return TEST_FAIL;
        }

        // verify the result
        size_t pat_ind = 0;
        for (auto&& pass : simul_passes)
        {
            auto& res_data = pass.output_buffer;
            for (size_t i = 0; i < num_elements; i++, pat_ind++)
            {
                CHECK_VERIFICATION_ERROR(sstr.str().at(pat_ind), res_data[i],
                                         i);
            }
        }

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    clEventWrapper trigger_event = nullptr;

    std::string temp_filename;
    int file_descriptor;

    // specifies max test length for printf pattern
    const unsigned max_pattern_length = 6;
    // specifies min test length for printf pattern
    const unsigned min_pattern_length = 1;
    // specifies number of command-buffer equeue iterations
    const unsigned num_test_iters = 3;
};

//#undef CHECK_VERIFICATION_ERROR

} // anonymous namespace

int test_basic_printf(cl_device_id device, cl_context context,
                                              cl_command_queue queue, int num_elements)
{
  return MakeAndRunTest<CommandBufferPrintfTest<false> >(device, context, queue, num_elements);
}

int test_simultaneous_printf(cl_device_id device, cl_context context,
                                                cl_command_queue queue, int num_elements)
{
  return MakeAndRunTest<CommandBufferPrintfTest<true> >(device, context, queue, num_elements);
}

