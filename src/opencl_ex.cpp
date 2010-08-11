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
unsigned int BLOCK_SIZE = 512;
const unsigned int N=4*1000;

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

void search_cpu(int len,int N, float *value_x, float* value_y, int *index_cpu, int *level_list, int* leaf_list, float* centerx_list, float *centery_list){
       for (int i = 0; i< len; i++){
           float width = pow(2.0,-level_list[i]);   
           float xmin = centerx_list[i] - width;
           float ymin = centery_list[i] - width;
           float xmax = centerx_list[i] + width;
           float ymax = centery_list[i] + width;
          for (int j=0;j<N;j++){
           if (value_x[j] >= xmin && value_x[j]<=xmax &&
            value_y[j]>ymin && value_y[j]<=ymax)
              //index_cpu[j] = leaf_list[i];
              index_cpu[j] = i ;
          }
       }
}

void interpolation_cpu(int N, float* value_x, float *value_y, int* index_g, int *level_list_d, float *centerx_list_d, float* centery_list_d,  float *T1_list_d, float* T2_list_d, float * T3_list_d, float* T4_list_d,float* interp_value){
    std::cout << "interpolation on cpu!"<<std::endl;
    for(int i = 0;i< N;i++){
	    int j = index_g[i];
	    float width = powf(2.0,-level_list_d[j]);
	    float xmin = centerx_list_d[j] - width;
	    float ymin = centery_list_d[j] - width;
	    float xmax = centerx_list_d[j] + width;
	    float ymax = centery_list_d[j] + width; 

	    // rescale x,y in the local cell
	    float x_ref = (value_x[i]-xmin)/(xmax-xmin);
	    float y_ref = (value_y[i]-ymin)/(ymax-xmin);
	   
	    // pickup the interpolation triangle 
	    float x_nodes[3], y_nodes[3], var[3];
	    x_nodes[0] = xmin;
	    x_nodes[1] = x_ref>=y_ref?  xmax: xmax ;
	    x_nodes[2] = x_ref>=y_ref?  xmax: xmin;

	    y_nodes[0] = ymin;
	    y_nodes[1] = x_ref>=y_ref? ymin:ymax ;
	    y_nodes[2] = x_ref>=y_ref? ymax:ymax ;
	   
	    var[0] = T1_list_d[j];
	    var[1] = x_ref>=y_ref? T2_list_d[j]: T3_list_d[j] ;
	    var[2] = x_ref>=y_ref? T3_list_d[j]: T4_list_d[j];

	float A = y_nodes[0]*(var[1]- var[2])  +  y_nodes[1]*(var[2] - var[0]) +  y_nodes[2]*(var[0] - var[1]);

	float B = var[0]*(x_nodes[1] - x_nodes[2]) + var[1]*(x_nodes[2] - x_nodes[0]) +  var[2]*(x_nodes[0] - x_nodes[1]);

	float C = x_nodes[0]*(y_nodes[1] - y_nodes[2]) + x_nodes[1]*(y_nodes[2] - y_nodes[0]) + x_nodes[2]*(y_nodes[0] - y_nodes[1]);

	float D = -A*x_nodes[0] - B*y_nodes[0] - C*var[0];
	interp_value[i] = -(A*value_x[i] + B*value_y[i] + D)/C;

   }
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
  cl_kernel       kernel[2]={NULL, NULL};
  cl_mem          c_data;
  cl_build_status build_status;
  cl_int         *h_data;

  int num_nodes, num_leafs;
  float rootwidth, xmin, xmax, ymin, ymax;
  int *level_list, *leaf_list;
  float *centerx_list, *centery_list; //*width_cpu;
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

  //queue = clCreateCommandQueue(context, device_id,
  //                             (cl_command_queue_properties) 0, &err);
  queue = clCreateCommandQueue(context, device_id,
                               CL_QUEUE_PROFILING_ENABLE, &err);
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

    clock_t starttime, endtime; 
    starttime = clock();
    search_cpu(size, N, value_x, value_y, index_cpu, level_list, leaf_list, centerx_list, centery_list);
    interpolation_cpu(N, value_x, value_y, index_cpu, level_list,centerx_list, centery_list, T1_list, T2_list, T3_list, T4_list,interp);
    endtime = clock();
    
   // allocate device memory and data
      printf("allocating memory on GPU!\n");
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
   T1_list_d = clCreateBuffer(context, (cl_mem_flags) CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * size, T1_list, &err);
  OPENCL_CHECK_ERR(err);
   T2_list_d = clCreateBuffer(context, (cl_mem_flags) CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * size, T2_list, &err);
  OPENCL_CHECK_ERR(err);
   T3_list_d = clCreateBuffer(context, (cl_mem_flags) CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * size, T3_list, &err);
  OPENCL_CHECK_ERR(err);
   T4_list_d = clCreateBuffer(context, (cl_mem_flags) CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * size, T4_list, &err);
  OPENCL_CHECK_ERR(err);

   interp_d = clCreateBuffer(context, (cl_mem_flags) CL_MEM_COPY_HOST_PTR,
                          sizeof(float) * N, interp, &err);
  OPENCL_CHECK_ERR(err);
  //cl_mem width;
  //width_cpu = (float *)  malloc(size*sizeof(float));

  //width = clCreateBuffer(context, (cl_mem_flags) CL_MEM_COPY_HOST_PTR,
  //                        sizeof(float) * size, index, &err);
  OPENCL_CHECK_ERR(err);
  opencl_ex_ptr = (char *) opencl_ex;
  program = clCreateProgramWithSource(context, 1,
                                      (const char **) &opencl_ex_ptr,
                                      NULL, &err);
  OPENCL_CHECK_ERR(err);


  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL );
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
      void        *build_log = NULL;
      //char        *build_log = NULL;
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

  kernel[0] = clCreateKernel(program, "search_kernel", &err);
  OPENCL_CHECK_ERR(err);
  kernel[1] = clCreateKernel(program, "interpolation", &err);
  OPENCL_CHECK_ERR(err);
  

  size_t    local_work_size; 
  clGetKernelWorkGroupInfo(kernel[0], device_id, CL_KERNEL_WORK_GROUP_SIZE,
                          sizeof(size_t), &local_work_size, NULL);
  printf("maxmium size : %d\n", local_work_size);
  local_work_size = BLOCK_SIZE;
  size_t    global_work_size =  ((num_leafs -1 +local_work_size)/local_work_size)*local_work_size;
  printf("num_cells: %d , blocksize: %d, num_threads : %d\n", num_leafs, local_work_size, global_work_size);
  OPENCL_SAFE_CALL(clSetKernelArg(kernel[0], 0, sizeof(int), &num_leafs));
  OPENCL_SAFE_CALL(clSetKernelArg(kernel[0], 1, sizeof( unsigned int), &N));
  OPENCL_SAFE_CALL(clSetKernelArg(kernel[0], 5, sizeof(level_list_d), &level_list_d));
  OPENCL_SAFE_CALL(clSetKernelArg(kernel[0], 6, sizeof(leaf_list_d), &leaf_list_d));
  OPENCL_SAFE_CALL(clSetKernelArg(kernel[0], 7, sizeof(centerx_list_d), &centerx_list_d));
  OPENCL_SAFE_CALL(clSetKernelArg(kernel[0], 8, sizeof(centery_list_d), &centery_list_d));
  OPENCL_SAFE_CALL(clSetKernelArg(kernel[0], 2, sizeof(value_x_d), &value_x_d));
  OPENCL_SAFE_CALL(clSetKernelArg(kernel[0], 3, sizeof(value_y_d), &value_y_d));
  OPENCL_SAFE_CALL(clSetKernelArg(kernel[0], 4, sizeof(index_g), &index_g));
  //OPENCL_SAFE_CALL(clSetKernelArg(kernel[0], 9, sizeof(width), &width));
  
  OPENCL_SAFE_CALL(clSetKernelArg(kernel[1], 0, sizeof(unsigned int), &N));
  OPENCL_SAFE_CALL(clSetKernelArg(kernel[1], 4, sizeof(level_list_d), &level_list_d));
  OPENCL_SAFE_CALL(clSetKernelArg(kernel[1], 5, sizeof(centerx_list_d), &centerx_list_d));
  OPENCL_SAFE_CALL(clSetKernelArg(kernel[1], 6, sizeof(centery_list_d), &centery_list_d));
  OPENCL_SAFE_CALL(clSetKernelArg(kernel[1], 1, sizeof(value_x_d), &value_x_d));
  OPENCL_SAFE_CALL(clSetKernelArg(kernel[1], 2, sizeof(value_y_d), &value_y_d));
  OPENCL_SAFE_CALL(clSetKernelArg(kernel[1], 3, sizeof(index_g), &index_g));
  OPENCL_SAFE_CALL(clSetKernelArg(kernel[1], 7, sizeof(T1_list_d), &T1_list_d));
  OPENCL_SAFE_CALL(clSetKernelArg(kernel[1], 8, sizeof(T2_list_d), &T2_list_d));
  OPENCL_SAFE_CALL(clSetKernelArg(kernel[1], 9, sizeof(T3_list_d), &T3_list_d));
  OPENCL_SAFE_CALL(clSetKernelArg(kernel[1], 10, sizeof(T4_list_d), &T4_list_d));
  OPENCL_SAFE_CALL(clSetKernelArg(kernel[1], 11, sizeof(interp_d), &interp_d));
  
  cl_event event;
  cl_ulong start;
  cl_ulong end;
  OPENCL_SAFE_CALL(clEnqueueNDRangeKernel
                       (queue, kernel[0], 1, NULL,  &global_work_size, &local_work_size, 0,
                        NULL, &event)
      );
  clWaitForEvents(1, &event);
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start,
  NULL);
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
  float search_time = (end - start)/1000000.0; 
  printf("pass the search_kernel!!\n");
 #if 1 
  OPENCL_SAFE_CALL(clEnqueueNDRangeKernel
                       (queue, kernel[1], 1, NULL,  &global_work_size, &local_work_size, 0,
                        NULL, &event)
                  );
  clWaitForEvents(1, &event);
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start,
  NULL);
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
#endif
  float interpolation_time = (end - start)/1000000.0; 
  OPENCL_SAFE_CALL(clEnqueueReadBuffer
                       (queue, index_g, CL_TRUE, 0, sizeof(int) * N,
                        index, 0, NULL, NULL)
                  );
  printf("pass the interpolation_kernel!!\n");
#if 1 
  OPENCL_SAFE_CALL(clEnqueueReadBuffer
                       (queue, interp_d, CL_TRUE, 0, sizeof(float) * N,
                        interp_h, 0, NULL, NULL)
                  );
#endif
    // check the result
    for (int i=0; i < N; i++){
       assert(interp[i]=interp_h[i]);
    //if (index[i]<0)
       //cout <<"cell %d is not in this range!, cpu: "<< i << " " <<index_cpu[i]<< endl;
     //  printf("the value is cell : %d %d \n",index[i],index_cpu[i] ); 
    //else
       //printf("the value is cell : %d %d \n",index[i],index_cpu[i] ); 
       //printf("the value is cell : %d %d %f %f\n",index[i],index_cpu[i], interp_h[i], interp[i] ); 
    }
    
    //output the time
    // printf("GPU %.1f ms\n", time);
    printf(" GPU: search_time:%10.5f [ms], interpolation time:%10.5f[ms], total time: %10.5f\n", search_time, interpolation_time, search_time + interpolation_time); 
     printf("CPU %ld ms\n", (int) (1000.0f * (endtime - starttime) / CLOCKS_PER_SEC));

  /* --------------clean up------------------*/
  OPENCL_SAFE_CALL(clReleaseMemObject(index_g));
  OPENCL_SAFE_CALL(clReleaseMemObject(leaf_list_d));
  OPENCL_SAFE_CALL(clReleaseMemObject(level_list_d));
  OPENCL_SAFE_CALL(clReleaseMemObject(centery_list_d));
  OPENCL_SAFE_CALL(clReleaseMemObject(centerx_list_d));
  OPENCL_SAFE_CALL(clReleaseMemObject(value_x_d));
  OPENCL_SAFE_CALL(clReleaseMemObject(value_y_d));
  OPENCL_SAFE_CALL(clReleaseMemObject(T1_list_d));
 OPENCL_SAFE_CALL(clReleaseMemObject(T2_list_d));
  OPENCL_SAFE_CALL(clReleaseMemObject(T3_list_d));
  OPENCL_SAFE_CALL(clReleaseMemObject(T4_list_d));
  OPENCL_SAFE_CALL(clReleaseMemObject(interp_d));
 
  OPENCL_SAFE_CALL(clReleaseKernel(kernel[0]));
  OPENCL_SAFE_CALL(clReleaseKernel(kernel[1]));
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
	free(T1_list);
        free(T2_list);
	free(T3_list);
	free(T4_list);
	free(interp_h);
	free(index_cpu);
  return failures;
}
