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
#include "procs.h"

#include <vector>

namespace {

////////////////////////////////////////////////////////////////////////////////
// Command-queue substitution tests which handles below cases:
// -substitution on queue without properties
// -substitution on queue with properties
// -simultaneous use queue substitution

template < bool simul_use >
struct CommandBufferPrintfTest : public BasicCommandBufferTest
{
    CommandBufferPrintfTest(cl_device_id device, cl_context context,
                        cl_command_queue queue)
        : BasicCommandBufferTest(device, context, queue),
          user_event(nullptr)
    {
        double_buffers_size = simultaneous_use_requested = simul_use;
    }

    //--------------------------------------------------------------------------
    bool Skip() override
    {
      return (simultaneous_use_requested && !simultaneous_use)
          || BasicCommandBufferTest::Skip();
    }

    //--------------------------------------------------------------------------
    cl_int CreateCommandQueueWithProperties(cl_command_queue &queue_with_prop)
    {
        cl_int error = 0;
        cl_queue_properties_khr device_props = 0;

        error = clGetDeviceInfo(device, CL_DEVICE_QUEUE_PROPERTIES,
                                sizeof(device_props), &device_props, nullptr);
        test_error(error,
                   "clGetDeviceInfo for CL_DEVICE_QUEUE_PROPERTIES failed");

        using PropPair = std::pair<cl_queue_properties_khr, std::string>;

        auto check_property = [&](const PropPair & prop)
        {
          if (device_props & prop.first)
          {
            log_info("Queue property %s supported. Testing ... \n",
                     prop.second.c_str());
            queue_with_prop = clCreateCommandQueue
                (context, device, prop.first, &error);
          }
          else
              log_info("Queue property %s not supported \n", prop.second.c_str());
        };

        // in case of extending property list in future
        std::vector< PropPair > props = {
          ADD_PROP(CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE),
          ADD_PROP(CL_QUEUE_PROFILING_ENABLE)
        };

        for ( auto && prop : props )
        {
          check_property(prop);
          test_error(error, "clCreateCommandQueue failed");
          if (queue_with_prop!=nullptr)
              return CL_SUCCESS;
        }

        return CL_INVALID_QUEUE_PROPERTIES;
    }

    //--------------------------------------------------------------------------
    cl_int SetUp(int elements) override
    {
        // TODO : setup command buffer with special printf kernel
        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int Run() override
    {
        cl_int error = CL_SUCCESS;
        cl_command_queue_properties cqp;
        error = clGetCommandQueueInfo(queue, CL_QUEUE_PROPERTIES, sizeof(cqp), &cqp, NULL);
        test_error(error, "clGetCommandQueueInfo failed");

        // record command buffer with primary queue
        error = RecordCommandBuffer();
        test_error(error, "RecordCommandBuffer failed");

        if (simultaneous_use)
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
    cl_int RunSingle()
    {
      cl_int error = CL_SUCCESS;
      std::vector<cl_int> output_data(num_elements);

      error = clEnqueueFillBuffer(queue, in_mem, &pattern_pri, sizeof(cl_int),
                                  0, data_size(), 0, nullptr, nullptr);
      test_error(error, "clEnqueueFillBuffer failed");

      error = clEnqueueCommandBufferKHR(0, nullptr, command_buffer, 0,
                                        nullptr, nullptr);
      test_error(error, "clEnqueueCommandBufferKHR failed");

      error = clEnqueueReadBuffer(queue, out_mem, CL_TRUE, 0, data_size(),
                                  output_data.data(), 0, nullptr, nullptr);
      test_error(error, "clEnqueueReadBuffer failed");

      error = clFinish(queue);
      test_error(error, "clFinish failed");

      for (size_t i = 0; i < num_elements; i++)
      {
          CHECK_VERIFICATION_ERROR(pattern_pri, output_data[i], i);
      }

      return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    // tuple order: pattern, offset, queue, output-buffer
    using SimulPassData =
      std::tuple<cl_int, cl_int, cl_command_queue, std::vector<cl_int>>;

    //--------------------------------------------------------------------------
    cl_int EnqueueSimultaneousPass (SimulPassData & pd)
    {
      const cl_int offset = std::get<1>(pd);
      auto & q = std::get<2>(pd);
      cl_int error = clEnqueueFillBuffer
          (q, in_mem, &std::get<0>(pd), sizeof(cl_int),
           offset * sizeof(cl_int), data_size(), 0, nullptr, nullptr);
      test_error(error, "clEnqueueFillBuffer failed");

#if USE_COMMAND_BUF_KENEL_ARG
      error = clSetKernelArg(kernel, 2, sizeof(cl_int), &offset);
      test_error(error, "clSetKernelArg failed");
#else
         error = clEnqueueFillBuffer
             (q, off_mem, &offset, sizeof(cl_int), 0, sizeof(cl_int),
              0, nullptr, nullptr);
         test_error(error, "clEnqueueFillBuffer failed");

#endif

      if (!user_event)
      {
        user_event = clCreateUserEvent(context, &error);
        test_error(error, "clCreateUserEvent failed");
      }

      cl_command_queue queues[] = { q };
      error = clEnqueueCommandBufferKHR
          (1, queues, command_buffer, 1, &user_event, nullptr);
      test_error(error, "clEnqueueCommandBufferKHR failed");

      error = clEnqueueReadBuffer
          (q, out_mem, CL_FALSE, offset * sizeof(cl_int),
           data_size(), std::get<3>(pd).data(), 0, nullptr, nullptr);

      test_error(error, "clEnqueueReadBuffer failed");

      return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    cl_int RunSimultaneous()
    {
        cl_int error = CL_SUCCESS;

        // tuple order: pattern, offset, queue, output-buffer
        std::vector<SimulPassData> simul_passes = {
          { pattern_pri, 0, queue, std::vector<cl_int>(num_elements) },
          { pattern_sec, num_elements, queue, std::vector<cl_int>(num_elements) }
        };

        for ( auto && pass : simul_passes )
        {
          error = EnqueueSimultaneousPass(pass);
          test_error(error, "EnqueuePass failed");
        }

        error = clSetUserEventStatus(user_event, CL_COMPLETE);
        test_error(error, "clSetUserEventStatus failed");

        for ( auto && pass : simul_passes )
        {
          error = clFinish(std::get<2>(pass));
          test_error(error, "clFinish failed");

          auto & pattern = std::get<0>(pass);
          auto & res_data = std::get<3>(pass);

          for (size_t i = 0; i < num_elements; i++)
          {
              CHECK_VERIFICATION_ERROR(pattern, res_data[i], i);
          }
        }

        return CL_SUCCESS;
    }

    //--------------------------------------------------------------------------
    const cl_int pattern_pri = 0xB;
    const cl_int pattern_sec = 0xC;

    clEventWrapper user_event = nullptr;
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
