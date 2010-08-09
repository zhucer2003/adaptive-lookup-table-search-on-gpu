#include <config.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#ifdef HAVE_OPENCL
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#else
#error "Needs OpenCL"
#endif

#define OPENCL_CHECK_ERR(err) do {                        \
    if ( err != CL_SUCCESS ) {                            \
      char cl_err_string[BUFSIZ];                         \
      cl_get_err_string(err, BUFSIZ, cl_err_string);      \
      fprintf(stderr, "OpenCL Error: %s", cl_err_string); \
      exit(-1);                                           \
    }                                                     \
 } while (0)
#define OPENCL_SAFE_CALL(call) do { \
    cl_int cl_err = call;           \
    OPENCL_CHECK_ERR(cl_err);       \
 } while (0)

#include <opencl_ex.h>

#define SIZE 34

void
cl_get_err_string(cl_int err, size_t n, char *str)
{
  switch (err) {
  case CL_SUCCESS:
    strncpy(str, "CL_SUCCESS", n);
    break;
  case CL_DEVICE_NOT_FOUND:
    strncpy(str, "CL_DEVICE_NOT_FOUND", n);
    break;
  case CL_DEVICE_NOT_AVAILABLE:
    strncpy(str, "CL_DEVICE_NOT_AVAILABLE", n);
    break;
  case CL_COMPILER_NOT_AVAILABLE:
    strncpy(str, "CL_COMPILER_NOT_AVAILABLE", n);
    break;
  case CL_MEM_OBJECT_ALLOCATION_FAILURE:
    strncpy(str, "CL_MEM_OBJECT_ALLOCATION_FAILURE", n);
    break;
  case CL_OUT_OF_RESOURCES:
    strncpy(str, "CL_OUT_OF_RESOURCES", n);
    break;
  case CL_OUT_OF_HOST_MEMORY:
    strncpy(str, "CL_OUT_OF_HOST_MEMORY", n);
    break;
  case CL_PROFILING_INFO_NOT_AVAILABLE:
    strncpy(str, "CL_PROFILING_INFO_NOT_AVAILABLE", n);
    break;
  case CL_MEM_COPY_OVERLAP:
    strncpy(str, "CL_MEM_COPY_OVERLAP", n);
    break;
  case CL_IMAGE_FORMAT_MISMATCH:
    strncpy(str, "CL_IMAGE_FORMAT_MISMATCH", n);
    break;
  case CL_IMAGE_FORMAT_NOT_SUPPORTED:
    strncpy(str, "CL_IMAGE_FORMAT_NOT_SUPPORTED", n);
    break;
  case CL_BUILD_PROGRAM_FAILURE:
    strncpy(str, "CL_BUILD_PROGRAM_FAILURE", n);
    break;
  case CL_MAP_FAILURE:
    strncpy(str, "CL_MAP_FAILURE", n);
    break;
#ifdef CL_VERSION_1_1
  case CL_MISALIGNED_SUB_BUFFER_OFFSET:
    strncpy(str, "CL_MISALIGNED_SUB_BUFFER_OFFSET", n);
    break;
  case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
    strncpy(str, "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST", n);
    break;
#endif

  case CL_INVALID_VALUE:
    strncpy(str, "CL_INVALID_VALUE", n);
    break;
  case CL_INVALID_DEVICE_TYPE:
    strncpy(str, "CL_INVALID_DEVICE_TYPE", n);
    break;
  case CL_INVALID_PLATFORM:
    strncpy(str, "CL_INVALID_PLATFORM", n);
    break;
  case CL_INVALID_DEVICE:
    strncpy(str, "CL_INVALID_DEVICE", n);
    break;
  case CL_INVALID_CONTEXT:
    strncpy(str, "CL_INVALID_CONTEXT", n);
    break;
  case CL_INVALID_QUEUE_PROPERTIES:
    strncpy(str, "CL_INVALID_QUEUE_PROPERTIES", n);
    break;
  case CL_INVALID_COMMAND_QUEUE:
    strncpy(str, "CL_INVALID_COMMAND_QUEUE", n);
    break;
  case CL_INVALID_HOST_PTR:
    strncpy(str, "CL_INVALID_HOST_PTR", n);
    break;
  case CL_INVALID_MEM_OBJECT:
    strncpy(str, "CL_INVALID_MEM_OBJECT", n);
    break;
  case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
    strncpy(str, "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR", n);
    break;
  case CL_INVALID_IMAGE_SIZE:
    strncpy(str, "CL_INVALID_IMAGE_SIZE", n);
    break;
  case CL_INVALID_SAMPLER:
    strncpy(str, "CL_INVALID_SAMPLER", n);
    break;
  case CL_INVALID_BINARY:
    strncpy(str, "CL_INVALID_BINARY", n);
    break;
  case CL_INVALID_BUILD_OPTIONS:
    strncpy(str, "CL_INVALID_BUILD_OPTIONS", n);
    break;
  case CL_INVALID_PROGRAM:
    strncpy(str, "CL_INVALID_PROGRAM", n);
    break;
  case CL_INVALID_PROGRAM_EXECUTABLE:
    strncpy(str, "CL_INVALID_PROGRAM_EXECUTABLE", n);
    break;
  case CL_INVALID_KERNEL_NAME:
    strncpy(str, "CL_INVALID_KERNEL_NAME", n);
    break;
  case CL_INVALID_KERNEL_DEFINITION:
    strncpy(str, "CL_INVALID_KERNEL_DEFINITION", n);
    break;
  case CL_INVALID_KERNEL:
    strncpy(str, "CL_INVALID_KERNEL", n);
    break;
  case CL_INVALID_ARG_INDEX:
    strncpy(str, "CL_INVALID_ARG_INDEX", n);
    break;
  case CL_INVALID_ARG_VALUE:
    strncpy(str, "CL_INVALID_ARG_VALUE", n);
    break;
  case CL_INVALID_ARG_SIZE:
    strncpy(str, "CL_INVALID_ARG_SIZE", n);
    break;
  case CL_INVALID_KERNEL_ARGS:
    strncpy(str, "CL_INVALID_KERNEL_ARGS", n);
    break;
  case CL_INVALID_WORK_DIMENSION:
    strncpy(str, "CL_INVALID_WORK_DIMENSION", n);
    break;
  case CL_INVALID_WORK_GROUP_SIZE:
    strncpy(str, "CL_INVALID_WORK_GROUP_SIZE", n);
    break;
  case CL_INVALID_WORK_ITEM_SIZE:
    strncpy(str, "CL_INVALID_WORK_ITEM_SIZE", n);
    break;
  case CL_INVALID_GLOBAL_OFFSET:
    strncpy(str, "CL_INVALID_GLOBAL_OFFSET", n);
    break;
  case CL_INVALID_EVENT_WAIT_LIST:
    strncpy(str, "CL_INVALID_EVENT_WAIT_LIST", n);
    break;
  case CL_INVALID_EVENT:
    strncpy(str, "CL_INVALID_EVENT", n);
    break;
  case CL_INVALID_OPERATION:
    strncpy(str, "CL_INVALID_OPERATION", n);
    break;
  case CL_INVALID_GL_OBJECT:
    strncpy(str, "CL_INVALID_GL_OBJECT", n);
    break;
  case CL_INVALID_BUFFER_SIZE:
    strncpy(str, "CL_INVALID_BUFFER_SIZE", n);
    break;
  case CL_INVALID_MIP_LEVEL:
    strncpy(str, "CL_INVALID_MIP_LEVEL", n);
    break;
  case CL_INVALID_GLOBAL_WORK_SIZE:
    strncpy(str, "CL_INVALID_GLOBAL_WORK_SIZE", n);
    break;

  default:
    strncpy(str, "Unknown", n);
  }

  if (n > 0)
    str[n - 1] = '\0';

  return;
}

int
main(int argc, char *argv[])
{
  int             failures = 0;

  cl_uint         i;
  const cl_uint   MAX_PLATFORMS = 256;
  cl_platform_id  platforms[MAX_PLATFORMS];
  cl_uint         num_platforms;
  cl_device_id    device_id;
  cl_context      context;
  cl_int          err;
  char           *opencl_ex_ptr;

  cl_command_queue queue;
  cl_program      program;
  cl_kernel       kernel;
  cl_mem          c_data;
  cl_build_status build_status;
  cl_int         *h_data;
  size_t          global_work_size[] = { SIZE, 0, 0 };

  OPENCL_SAFE_CALL(clGetPlatformIDs(MAX_PLATFORMS, platforms,
                                        &num_platforms));

  for (i = 0; i < num_platforms; ++i) {
    char            buf[BUFSIZ];

    printf("%3d: ", (int) i);

    OPENCL_SAFE_CALL(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME,
                                           sizeof(buf), buf, NULL));

    printf("%s - ", buf);

    OPENCL_SAFE_CALL(clGetPlatformInfo(platforms[i],
                                           CL_PLATFORM_VERSION,
                                           sizeof(buf), buf, NULL));

    printf("%s - ", buf);

    OPENCL_SAFE_CALL(clGetPlatformInfo(platforms[i],
                                           CL_PLATFORM_VENDOR,
                                           sizeof(buf), buf, NULL));

    printf("%s - ", buf);

    OPENCL_SAFE_CALL(clGetPlatformInfo(platforms[i],
                                           CL_PLATFORM_PROFILE,
                                           sizeof(buf), buf, NULL));

    printf("%s\n", buf);
  }

  h_data = (cl_int *) calloc(SIZE, sizeof(cl_int));
  for (i = 0; i < SIZE; ++i) {
    h_data[i] = i;
  }

  assert(num_platforms > 0);

  OPENCL_SAFE_CALL(clGetDeviceIDs(platforms[0], (cl_device_type)
                                      CL_DEVICE_TYPE_DEFAULT, 1,
                                      &device_id, NULL));

  context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
  OPENCL_CHECK_ERR(err);

  queue = clCreateCommandQueue(context, device_id,
                               (cl_command_queue_properties) 0, &err);
  OPENCL_CHECK_ERR(err);

  c_data = clCreateBuffer(context, (cl_mem_flags) CL_MEM_COPY_HOST_PTR,
                          sizeof(cl_int) * SIZE, h_data, &err);
  OPENCL_CHECK_ERR(err);

  opencl_ex_ptr = (char *) opencl_ex;
  program = clCreateProgramWithSource(context, 1,
                                      (const char **) &opencl_ex_ptr,
                                      NULL, &err);
  OPENCL_CHECK_ERR(err);

  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err != CL_SUCCESS && err != CL_BUILD_PROGRAM_FAILURE) {
    OPENCL_CHECK_ERR(err);
  }
  do {
    /*
     * Check to see if the compile was sucessful
     */
    OPENCL_CHECK_ERR(clGetProgramBuildInfo
                         (program, device_id, CL_PROGRAM_BUILD_STATUS,
                          sizeof(cl_build_status), &build_status, NULL)
        );

    if (build_status == CL_BUILD_ERROR) {
      /*
       * Print the build log
       */
      cl_char        *build_log = NULL;
      size_t          build_log_len = 0;

      OPENCL_CHECK_ERR(clGetProgramBuildInfo
                           (program, device_id, CL_PROGRAM_BUILD_LOG, 0,
                            NULL, &build_log_len)
          );

      build_log = malloc(build_log_len + 1);

      OPENCL_CHECK_ERR(clGetProgramBuildInfo
                           (program, device_id, CL_PROGRAM_BUILD_LOG,
                            build_log_len, build_log, NULL));
      fprintf(stderr, "OpenCL program failed to build.\n%s", build_log);

      free(build_log);
    }

  } while (build_status == CL_BUILD_IN_PROGRESS);

  kernel = clCreateKernel(program, "square", &err);
  OPENCL_CHECK_ERR(err);

  OPENCL_SAFE_CALL(clSetKernelArg(kernel, 0, sizeof(c_data), &c_data));
  OPENCL_SAFE_CALL(clEnqueueNDRangeKernel
                       (queue, kernel, 1, NULL, global_work_size, NULL, 0,
                        NULL, NULL)
      );

  OPENCL_SAFE_CALL(clEnqueueReadBuffer
                       (queue, c_data, CL_TRUE, 0, sizeof(cl_int) * SIZE,
                        h_data, 0, NULL, NULL)
      );

  for (i = 0; i < SIZE; ++i) {
    printf("%d: %d\n", (int) i, (int) h_data[i]);
    if (h_data[i] != (cl_int) (i * i))
      ++failures;
  }


  OPENCL_SAFE_CALL(clReleaseMemObject(c_data));
  OPENCL_SAFE_CALL(clReleaseKernel(kernel));
  OPENCL_SAFE_CALL(clReleaseProgram(program));
  OPENCL_SAFE_CALL(clReleaseCommandQueue(queue));
  OPENCL_SAFE_CALL(clReleaseContext(context));
  free(h_data);

  return failures;
}
