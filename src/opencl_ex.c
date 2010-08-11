#include <config.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <time.h>

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

#define MULT 1103515245
#define ADD 12345
#define MASK 0x7FFFFFFF
#define TWOTO31 2147483648.0

static int A = 1;
static int B = 0;
static int randx = 1;
static int lastrand;


static void drndset(int seed)
{
   A = 1;
   B = 0;
   randx = (A * seed + B) & MASK;
   A = (MULT * A) & MASK;
   B = (MULT * B + ADD) & MASK;
}


static double drnd()
{
   lastrand = randx;
   randx = (A * randx + B) & MASK;
   return (double)lastrand / TWOTO31;
}

#define SIZE 34

#define N 10*1000;

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

int main(int argc, char *argv[])
{

  using namespace std;
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

  int num_nodes, num_leafs;
  float rootwidth, xmin, xmax, ymin, ymax;
  int *level_list, *leaf_list;
  float *centerx_list, *centery_list;
  float *T1_list, *T2_list, *T3_list, *T4_list,*P1_list, *P2_list,*P3_list, *P4_list; // variable lists
        

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
  
  // read the data

        ifstream myfile("RPTBDB.dat");
        myfile >> num_nodes;
        myfile >> ymin >> ymax >> xmin >> xmax;
        myfile >> num_leafs >> rootwidth >> rootwidth;
        
        unsigned int bytes; 
        int fbytes = num_leafs*sizeof(float);
 	bytes = num_leafs * sizeof(int);
	int dbytes = sizeof(float);
	
        level_list = (int *) malloc( bytes);
	leaf_list = (int  *) malloc( bytes);
	centerx_list = (float *) malloc( fbytes);
	centery_list= (float *) malloc( fbytes);
	T1_list = (float *) malloc( fbytes);
	T2_list = (float *) malloc( fbytes);
	T3_list= (float *) malloc( fbytes);
	T4_list= (float *) malloc( fbytes);
	P1_list = (float *) malloc( fbytes);
	P2_list = (float *) malloc( fbytes);
	P3_list= (float *) malloc( fbytes);
	P4_list= (float *) malloc( fbytes);
        if (myfile.is_open())
	{
	  for(int i=0;i< num_leafs; i++){
	     myfile >> level_list[i] >> leaf_list[i];
	     myfile >> centerx_list[i] >> centery_list[i];
	     myfile >> T1_list[i] >> P1_list[i];
	     myfile >> T2_list[i] >> P2_list[i];
	     myfile >> T3_list[i] >> P3_list[i];
	     myfile >> T4_list[i] >> P4_list[i];
             }
	}
	myfile.close();

	int size= num_leafs; // numbet of elements to reduce 

  assert(num_platforms > 0);

  OPENCL_SAFE_CALL(clGetDeviceIDs(platforms[0], (cl_device_type)
                                      CL_DEVICE_TYPE_DEFAULT, 1,
                                      &device_id, NULL));

  context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
  OPENCL_CHECK_ERR(err);

  queue = clCreateCommandQueue(context, device_id,
                               (cl_command_queue_properties) 0, &err);
  OPENCL_CHECK_ERR(err);


	// allocate variables on GPU
	cl_mem level_list_d=NULL, leaf_list_d=NULL;
        cl_mem centerx_list_d=NULL, centery_list_d=NULL;
	cl_mem T1_list_d=NULL, T2_list_d=NULL, T3_list_d=NULL, T4_list_d=NULL;
        cl_mem index_g =NULL;
        cl_mem value_x_d=NULL, value_y_d=NULL, interp_d=NULL ;
        float *interp_h, *interp=NULL;
        int *index, *index_cpu;
        float *value_x=NULL, *value_y=NULL;

        //rescale the input the data 
        value_x = (float *) malloc( N*dbytes);
        value_y = (float *) malloc( N*dbytes);
        index = (int *) malloc( N*sizeof(int));
        interp_h = (float *) malloc( N*dbytes);
        interp = (float *) malloc( N*dbytes);

        drndset(9);
        index_cpu = (int *) malloc( N*sizeof(int));

        for (int i=0; i < N; i++){
		value_x[i] = drnd()*600 + 400;
		value_y[i] = drnd()*2.0 - 1.0;
		value_x[i] = (value_x[i]-xmin)/(xmax-xmin);
		value_y[i] = (value_y[i]-ymin)/(ymax-ymin);
                index[i] = -1;
                index_cpu[i]=-1;
                interp[i] = -1;
                interp_h[i] = -1;
                //cout << i << " " <<value_x[i] << " " << value_y[i]<<endl;
        }

        // allocate device memory and data
      printf("allocating memory on GPU!");
   level_list_d = clCreateBuffer(context, (cl_mem_flags) CL_MEM_COPY_HOST_PTR,
                          sizeof(int) * size, level_list, &err);
  OPENCL_CHECK_ERR(err);

   leaf_list_d = clCreateBuffer(context, (cl_mem_flags) CL_MEM_COPY_HOST_PTR,
                          sizeof(int) * size, leaf_list, &err);
  OPENCL_CHECK_ERR(err);
   
   centerx_list_d = clCreateBuffer(context, (cl_mem_flags) CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * size, centerx_list, &err);
  OPENCL_CHECK_ERR(err);
   
   centery_list_d = clCreateBuffer(context, (cl_mem_flags) CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * size, centery_list, &err);
  OPENCL_CHECK_ERR(err);
   
   value_x_d = clCreateBuffer(context, (cl_mem_flags) CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * N, value_x, &err);
  OPENCL_CHECK_ERR(err);
   value_y_d = clCreateBuffer(context, (cl_mem_flags) CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * N, value_y, &err);
  OPENCL_CHECK_ERR(err);
   
   index_g = clCreateBuffer(context, (cl_mem_flags) CL_MEM_COPY_HOST_PTR,
                          sizeof(int) * N, index, &err);
  OPENCL_CHECK_ERR(err);
  
  opencl_ex_ptr = (char *) opencl_ex;
  program = clCreateProgramWithSource(context, 1,
                                      (const char **) &opencl_ex_ptr,
                                      NULL, &err);
  OPENCL_CHECK_ERR(err);


  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
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

  kernel = clCreateKernel(program, "search_kernel", &err);
  OPENCL_CHECK_ERR(err);

  OPENCL_SAFE_CALL(clSetKernelArg(kernel, 0, sizeof(c_data), &c_data));
  OPENCL_SAFE_CALL(clEnqueueNDRangeKernel
                       (queue, kernel, 1, NULL, global_work_size, NULL, 0,
                        NULL, NULL)
      );

  OPENCL_SAFE_CALL(clEnqueueReadBuffer
                       (queue, index_g, CL_TRUE, 0, sizeof(int) * N,
                        index, 0, NULL, NULL)
      );

    // check the result
    for (int i=0; i < N; i++){
       assert(interp[i]=interp_h[i]);
    if (index[i]<0)
       printf("cell %d is not in this range!, cpu\n", i);
    
    //else
       //printf("the value is cell : %d %d \n",index[i],index_cpu[i] ); 
    //   printf("the value is cell : %d %d %f %f\n",index[i],index_cpu[i], interp_h[i], interp[i] ); 
    }

  /* --------------clean up------------------*/
  OPENCL_SAFE_CALL(clReleaseMemObject(index_g));
  OPENCL_SAFE_CALL(clReleaseMemObject(leaf_list_d));
  OPENCL_SAFE_CALL(clReleaseMemObject(level_list_d));
  OPENCL_SAFE_CALL(clReleaseMemObject(centery_list_d));
  OPENCL_SAFE_CALL(clReleaseMemObject(centerx_list_d));
  OPENCL_SAFE_CALL(clReleaseMemObject(value_x_d));
  OPENCL_SAFE_CALL(clReleaseKernel(value_y_d));
  OPENCL_SAFE_CALL(clReleaseProgram(program));
  OPENCL_SAFE_CALL(clReleaseCommandQueue(queue));
  OPENCL_SAFE_CALL(clReleaseContext(context));
  free(index);
        free(level_list);
	free(leaf_list);
	free(centerx_list);
	free(centery_list);
	free(value_x);
	free(value_y);
  return failures;
}
