//
// Copyright (c) 2023 The Khronos Group Inc.
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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "procs.h"

cl_int get_type_size( cl_context context, cl_command_queue queue, const char *type, cl_ulong *size, cl_device_id device  )
{
    const char *sizeof_kernel_code[4] =
    {
        "", /* optional pragma string */
        "__kernel __attribute__((reqd_work_group_size(1,1,1))) void test_sizeof(__global uint *dst) \n"
        "{\n"
        "   dst[0] = (uint) sizeof( ", type, " );\n"
        "}\n"
    };

    clProgramWrapper p;
    clKernelWrapper k;
    clMemWrapper m;
    cl_uint        temp;

    if (!strncmp(type, "double", 6))
        sizeof_kernel_code[0] = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
    else if (!strncmp(type, "half", 4))
        sizeof_kernel_code[0] = "#pragma OPENCL EXTENSION cl_khr_fp16 : enable\n";

    cl_int err = create_single_kernel_helper_with_build_options(
        context, &p, &k, 4, sizeof_kernel_code, "test_sizeof", nullptr);
    test_error(err, "Failed to build kernel/program.");

    m = clCreateBuffer( context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof( cl_ulong ), size, &err );
    test_error(err, "clCreateBuffer failed.");

    err = clSetKernelArg( k, 0, sizeof( cl_mem ), &m );
    test_error(err, "clSetKernelArg failed.");

    err = clEnqueueTask( queue, k, 0, NULL, NULL );
    test_error(err, "clEnqueueTask failed.");

    err = clEnqueueReadBuffer( queue, m, CL_TRUE, 0, sizeof( cl_uint ), &temp, 0, NULL, NULL );
    test_error(err, "clEnqueueReadBuffer failed.");

    *size = (cl_ulong) temp;
    return err;
}

typedef struct size_table
{
    const char *name;
    cl_ulong   size;
    cl_ulong   cl_size;
}size_table;

const size_table  scalar_table[] =
{
    // Fixed size entries from table 6.1
    {  "unsigned char",     1,  sizeof( cl_uchar)   },
    {  "unsigned short",    2,  sizeof( cl_ushort)  },
    {  "unsigned int",      4,  sizeof( cl_uint)    },
    {  "unsigned long",     8,  sizeof( cl_ulong)   }
};

const size_table vector_table[] = {
    // Fixed size entries from table 6.1
    { "char", 1, sizeof(cl_char) },     { "uchar", 1, sizeof(cl_uchar) },
    { "short", 2, sizeof(cl_short) },   { "ushort", 2, sizeof(cl_ushort) },
    { "int", 4, sizeof(cl_int) },       { "uint", 4, sizeof(cl_uint) },
    { "half", 2, sizeof(cl_half) },     { "float", 4, sizeof(cl_float) },
    { "double", 8, sizeof(cl_double) }, { "long", 8, sizeof(cl_long) },
    { "ulong", 8, sizeof(cl_ulong) }
};

const char  *ptr_table[] =
{
    "global void*",
    "size_t",
    "sizeof(int)",      // check return type of sizeof
    "ptrdiff_t"
};

const char *other_types[] =
{
    "event_t",
    "image2d_t",
    "image3d_t",
    "sampler_t"
};

static int IsPowerOfTwo( cl_ulong x ){ return 0 == (x & (x-1)); }

int test_sizeof(cl_device_id device, cl_context context, cl_command_queue queue, int num_elements)
{
    size_t i, j;
    cl_uint ptr_size = CL_UINT_MAX;

    // Check address space size
    cl_int err = clGetDeviceInfo(device, CL_DEVICE_ADDRESS_BITS,
                                 sizeof(ptr_size), &ptr_size, NULL);
    if( err || ptr_size > 64)
    {
        log_error( "FAILED:  Unable to get CL_DEVICE_ADDRESS_BITS for device %p\n", device );
        return -1;
    }
    log_info( "\tCL_DEVICE_ADDRESS_BITS = %u\n", ptr_size );
    ptr_size /= 8;

    // Test standard scalar sizes
    for( i = 0; i < sizeof( scalar_table ) / sizeof( scalar_table[0] ); i++ )
    {
        if (!gHasLong && 0 == strcmp(scalar_table[i].name, "unsigned long"))
        {
            log_info(
                "\nLongs are not supported by this device. Skipping test.\n");
            continue;
        }

        cl_ulong test = CL_ULONG_MAX;
        err = get_type_size( context, queue, scalar_table[i].name, &test, device);
        if( err )
            return err;
        if( test != scalar_table[i].size )
        {
            log_error( "\nFAILED: Type %s has size %lld, but expected size %lld!\n", scalar_table[i].name, test, scalar_table[i].size );
            return -1;
        }
        if( test != scalar_table[i].cl_size )
        {
            log_error( "\nFAILED: Type %s has size %lld, but cl_ size is %lld!\n", scalar_table[i].name, test, scalar_table[i].cl_size );
            return -2;
        }
        log_info( "%16s", scalar_table[i].name );
    }
    log_info( "\n" );

    bool hasFp64 = is_extension_available(device, "cl_khr_fp64");
    bool hasFp16 = is_extension_available(device, "cl_khr_fp16");
    const char *vec_size_names[] = { "", "2", "4", "8", "16" };

    // Test standard vector sizes
    for (j = 0; j < sizeof(vec_size_names) / sizeof(vec_size_names[0]); j++)
    {
        // For each vector size, iterate through types
        for( i = 0; i < sizeof( vector_table ) / sizeof( vector_table[0] ); i++ )
        {
            bool skip = false;
            if (!gHasLong
                && (0 == strcmp(vector_table[i].name, "long")
                    || 0 == strcmp(vector_table[i].name, "ulong")))
                skip = true;
            else if (!hasFp64 && 0 == strcmp(vector_table[i].name, "double"))
                skip = true;
            else if (!hasFp16 && 0 == strcmp(vector_table[i].name, "half"))
                skip = true;

            if (skip)
            {
                log_info(
                    "\n%s are not supported by this device. Skipping test.\n",
                    vector_table[i].name);
                continue;
            }

            char name[32];
            std::snprintf(name, sizeof(name), "%s%s", vector_table[i].name,
                          vec_size_names[j]);

            cl_ulong test = CL_ULONG_MAX;
            err = get_type_size( context, queue, name, &test, device  );
            test_error(err, "get_type_size failed");

            if (test != pow(2, j) * vector_table[i].size)
            {
                log_error("\nFAILED: Type %s has size %lld, but expected size "
                          "%lld!\n",
                          name, test, pow(2, j) * vector_table[i].size);
                return -1;
            }
            if (test != pow(2, j) * vector_table[i].cl_size)
            {
                log_error(
                    "\nFAILED: Type %s has size %lld, but cl_ size is %lld!\n",
                    name, test, pow(2, j) * vector_table[i].cl_size);
                return -2;
            }
            log_info( "%16s", name );
        }
        log_info( "\n" );
    }

    //Check that pointer sizes are correct
    for( i = 0; i < sizeof( ptr_table ) / sizeof( ptr_table[0] ); i++ )
    {
        cl_ulong test = CL_ULONG_MAX;
        err = get_type_size( context, queue, ptr_table[i], &test, device );
        test_error(err, "get_type_size failed");

        if( test != ptr_size )
        {
            log_error( "\nFAILED: Type %s has size %lld, but expected size %u!\n", ptr_table[i], test, ptr_size );
            return -1;
        }
        log_info( "%16s", ptr_table[i] );
    }

    auto test_pow2_type = [&](const char *type_name) {
        cl_ulong test = CL_ULONG_MAX;
        err = get_type_size(context, queue, type_name, &test, device);
        test_error(err, "get_type_size failed");

        if (test < ptr_size)
        {
            log_error("\nFAILED: %s has size %lld, but must be at least %u!\n",
                      type_name, test, ptr_size);
            return -1;
        }
        if (!IsPowerOfTwo(test))
        {
            log_error(
                "\nFAILED: sizeof(%s) is %lld, but must be a power of two!\n",
                type_name, test);
            return -2;
        }
        log_info("%16s", type_name);
        return 0;
    };

    // Check that intptr_t/uintptr_t is large enough
    err |= test_pow2_type("intptr_t");
    err |= test_pow2_type("uintptr_t");
    test_error(err, "test_pow2_type failed");

    //Check that other types are powers of two
    for( i = 0; i < sizeof( other_types ) / sizeof( other_types[0] ); i++ )
    {
        if( 0 == strcmp(other_types[i], "image2d_t") &&
           checkForImageSupport( device ) == CL_IMAGE_FORMAT_NOT_SUPPORTED)
        {
            log_info("\nimages are not supported by this device. Skipping test.\t");
            continue;
        }

        if (0 == strcmp(other_types[i], "image3d_t")
            && checkFor3DImageSupport(device) == CL_IMAGE_FORMAT_NOT_SUPPORTED)
        {
            log_info("\n3D images are not supported by this device. "
                     "Skipping test.\t");
            continue;
        }

        if( 0 == strcmp(other_types[i], "sampler_t") &&
           checkForImageSupport( device ) == CL_IMAGE_FORMAT_NOT_SUPPORTED)
        {
          log_info("\nimages are not supported by this device. Skipping test.\t");
          continue;
        }

        cl_ulong test = CL_ULONG_MAX;
        err = get_type_size( context, queue, other_types[i], &test, device );
        test_error(err, "get_type_size failed");

        if( ! IsPowerOfTwo( test ) )
        {
            log_error( "\nFAILED: Type %s has size %lld, which is not a power of two (section 6.1.5)!\n", other_types[i], test );
            return -1;
        }
        log_info( "%16s", other_types[i] );
    }
    log_info( "\n" );

    return err;
}
