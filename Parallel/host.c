/* Convolution example; originally written by Lucas Wilcox.
 * Minor modifications by Georg Stadler.
 * The function expects a bitmap image (*.ppm) as input, as
 * well as a number of blurring loops to be performed.
 */

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdbool.h>
#include "util.h"
#include "helper.h"
#include "cl-helper.h"

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

// local size of work group
#define WGX 16
#define WGY 16

float max(float a, float b){
  return a > b ? a : b;
}

float min(float a, float b){
  return a < b ? a : b;
}

void print_kernel_info(cl_command_queue queue, cl_kernel knl)
{
  // get device associated with the queue
  cl_device_id dev;
  CALL_CL_SAFE(clGetCommandQueueInfo(queue, CL_QUEUE_DEVICE,
        sizeof(dev), &dev, NULL));

  char kernel_name[4096];
  CALL_CL_SAFE(clGetKernelInfo(knl, CL_KERNEL_FUNCTION_NAME,
        sizeof(kernel_name), &kernel_name, NULL));
  kernel_name[4095] = '\0';
  printf("Info for kernel %s:\n", kernel_name);

  size_t kernel_work_group_size;
  CALL_CL_SAFE(clGetKernelWorkGroupInfo(knl, dev, CL_KERNEL_WORK_GROUP_SIZE,
        sizeof(kernel_work_group_size), &kernel_work_group_size, NULL));
  printf("  CL_KERNEL_WORK_GROUP_SIZE=%zd\n", kernel_work_group_size);

  size_t preferred_work_group_size_multiple;
  CALL_CL_SAFE(clGetKernelWorkGroupInfo(knl, dev,
        CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
        sizeof(preferred_work_group_size_multiple),
        &preferred_work_group_size_multiple, NULL));
  printf("  CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE=%zd\n",
      preferred_work_group_size_multiple);

  cl_ulong kernel_local_mem_size;
  CALL_CL_SAFE(clGetKernelWorkGroupInfo(knl, dev, CL_KERNEL_LOCAL_MEM_SIZE,
        sizeof(kernel_local_mem_size), &kernel_local_mem_size, NULL));
  printf("  CL_KERNEL_LOCAL_MEM_SIZE=%llu\n",
      (long long unsigned int)kernel_local_mem_size);

  cl_ulong kernel_private_mem_size;
  CALL_CL_SAFE(clGetKernelWorkGroupInfo(knl, dev, CL_KERNEL_PRIVATE_MEM_SIZE,
        sizeof(kernel_private_mem_size), &kernel_private_mem_size, NULL));
  printf("  CL_KERNEL_PRIVATE_MEM_SIZE=%llu\n",
      (long long unsigned int)kernel_private_mem_size);
}


int main(int argc, char *argv[])
{
   srand(time(NULL));
   // matrix sizes
   int n;
   int m;
   int i,j;

 //read the matrix from file
  FILE *fp;
  char *filename = "qtrain.txt";
  fp = fopen(filename,"r");
  if (fp == NULL) {
    printf("ERROR: unable to read file.\n");
    return -1;
  }

  char* line = NULL;
  size_t len = 0; //line length
  int lineLen = 0; //matrix length
  int lineNum = 0; //matrix height
  int passed = 0;

  //two passes, first pass to determine number of lines and line length
  // second pass to determine line length

  while (getline(&line,&len,fp) != -1) {
    if (passed == 0) {
      char* elts = strtok(line," ,\t");
      while (elts != NULL) {
        lineLen++;
        elts = strtok(NULL," ,\t");
      }
      passed = 1;
      free(elts);
    }
    lineNum++;
  }
  fclose(fp);

  //open again for pass 2
  fp = fopen(filename,"r");

  n = lineNum; // example size
  m = lineLen - 1;

  int Wy = 1;
  int Hy = n;
  int WX = m;
  int HX = n;
  unsigned int size_X = WX * HX;
  unsigned int mem_size_X = sizeof(float) * size_X;
  float* h_X = (float*) malloc(mem_size_X);

  unsigned int size_y = Wy * Hy;
  unsigned int mem_size_y = sizeof(float) * size_y;
  float* h_y = (float*) malloc(mem_size_y);

  for (i = 0;i<n;i++) {
    getline(&line,&len,fp);
    char* elts = strtok(line," ,\t");
    for (j=0;j<m;j++) {
      h_X[i * m + j] = strtof(elts,NULL);
      elts = strtok(NULL," ,\t");
    }
    h_y[i] = strtof(elts,NULL);
    elts = strtok(NULL," ,\t");
    free(elts);
  }
  fclose(fp);


  float* sum = calloc(m, sizeof(float));
  float* Max = calloc(m, sizeof(float));
  float* Min = calloc(m, sizeof(float));

  for (j = 0; j < m; j++){
    Max[j] = h_X[j];
    Min[j] = h_X[j]; 
  }
  
  // Normalize the data
  for (i = 0; i < n; i++){
    for (j = 0; j < m; j++){
      sum[j] += h_X[i * m + j];  
      Max[j] = max(Max[j], h_X[i * m + j]);
      Min[j] = min(Min[j], h_X[i * m + j]);  
    }
  }

  for (i = 0; i < m; i++){
    sum[i] /= n;
  } 

  for (i = 0; i < n; i++){
    for (j = 0; j < m; j++){
      h_X[i * m + j] = (h_X[i * m + j] - Min[j]) / (Max[j] - Min[j]);
      h_X[i * m + j] -= 0.5;
      h_X[i * m + j] /= 100;
    }
    h_y[i] /= 100;
  }

  int k = 5;
  // forward 1
  int WW1 = k;
  int HW1 = m;
  int Wz2 = k;
  int Hz2 = n;
  int Wa2 = k;
  int Ha2 = n;
  // forward 2
  int WW2 = 1;
  int HW2 = k;
  int Wz3 = 1;
  int Hz3 = n;
  int WyHat = 1;
  int HyHat = n;
  //cost func prime
  int Wdelta3 = 1;
  int Hdelta3 = n;
  int Wdelta2 = k;
  int Hdelta2 = n;
  int WdJdW2 = 1;
  int HdJdW2 = k;
  int WdJdW1 = k;
  int HdJdW1 = m;

  //Allocate host memory for matrices X and W and z2 and a2 and y
   
  
  unsigned int size_W1 = WW1 * HW1;
  unsigned int mem_size_W1 = sizeof(float) * size_W1;
  float* h_W1 = (float*) malloc(mem_size_W1);

  unsigned int size_z2 = Wz2 * Hz2;
  unsigned int mem_size_z2 = sizeof(float) * size_z2;
  float* h_z2 = (float*) malloc(mem_size_z2);
  
  unsigned int size_a2 = Wa2 * Ha2;
  unsigned int mem_size_a2 = sizeof(float) * size_a2;
  float* h_a2 = (float*) malloc(mem_size_a2);

  unsigned int size_W2 = WW2 * HW2;
  unsigned int mem_size_W2 = sizeof(float) * size_W2;
  float* h_W2 = (float*) malloc(mem_size_W2);

  unsigned int size_z3 = Wz3 * Hz3;
  unsigned int mem_size_z3 = sizeof(float) * size_z3;
  float* h_z3 = (float*) malloc(mem_size_z3);
  
  unsigned int size_yHat = WyHat * HyHat;
  unsigned int mem_size_yHat = sizeof(float) * size_yHat;
  float* h_yHat = (float*) malloc(mem_size_yHat);
   
  unsigned int size_delta3 = Wdelta3 * Hdelta3;
  unsigned int mem_size_delta3 = sizeof(float) * size_delta3;
  float* h_delta3 = (float*) malloc(mem_size_delta3);

  unsigned int size_delta2 = Wdelta2 * Hdelta2;
  unsigned int mem_size_delta2 = sizeof(float) * size_delta2;
  float* h_delta2 = (float*) malloc(mem_size_delta2);

  unsigned int size_dJdW2 = WdJdW2 * HdJdW2;
  unsigned int mem_size_dJdW2 = sizeof(float) * size_dJdW2;
  float* h_dJdW2 = (float*) malloc(mem_size_dJdW2);

  unsigned int size_dJdW1 = WdJdW1 * HdJdW1;
  unsigned int mem_size_dJdW1 = sizeof(float) * size_dJdW1;
  float* h_dJdW1 = (float*) malloc(mem_size_dJdW1);

  randomMemInit(h_W1, size_W1);
  randomMemInit(h_W2, size_W2);


  // --------------------------------------------------------------------------
  // get an OpenCL context and queue
  // --------------------------------------------------------------------------
  cl_context ctx;
  cl_command_queue queue;
  create_context_on(CHOOSE_INTERACTIVELY, CHOOSE_INTERACTIVELY, 0, &ctx, &queue, 0);
  print_device_info_from_queue(queue);

  // --------------------------------------------------------------------------
  // load kernels
  // --------------------------------------------------------------------------
  char *knl_text = read_file("NN_kernel.cl");
  cl_kernel knl1 = kernel_from_string(ctx, knl_text, "forward1", NULL);
  cl_kernel knl2 = kernel_from_string(ctx, knl_text, "forward2", NULL);
  cl_kernel knl3 = kernel_from_string(ctx, knl_text, "back1", NULL);
  cl_kernel knl4 = kernel_from_string(ctx, knl_text, "back2", NULL);
  cl_kernel knl5 = kernel_from_string(ctx, knl_text, "trans_dot", NULL);
  cl_kernel knl6 = kernel_from_string(ctx, knl_text, "update", NULL);
  cl_kernel knl7 = kernel_from_string(ctx, knl_text, "back3", NULL);
  free(knl_text);

  // --------------------------------------------------------------------------
  // allocate device memory
  // --------------------------------------------------------------------------
  cl_int status;
  cl_mem d_X = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
     mem_size_X, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  cl_mem d_W1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
      mem_size_W1, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  cl_mem d_z2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
     mem_size_z2, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  cl_mem d_a2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
     mem_size_a2, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  cl_mem d_W2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
     mem_size_W2, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  cl_mem d_z3 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
      mem_size_z3, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  cl_mem d_yHat = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
     mem_size_yHat, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  cl_mem d_y = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
     mem_size_y, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  cl_mem d_delta3 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
      mem_size_delta3, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  cl_mem d_dJdW2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
     mem_size_dJdW2, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  cl_mem d_delta2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
      mem_size_delta2, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  cl_mem d_dJdW1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE,
     mem_size_dJdW1, 0, &status);
  CHECK_CL_ERROR(status, "clCreateBuffer");

  CALL_CL_SAFE(clEnqueueWriteBuffer(
        queue, d_X, /*blocking*/ CL_TRUE, /*offset*/ 0,
        mem_size_X, h_X, 0, NULL, NULL));

  CALL_CL_SAFE(clEnqueueWriteBuffer(
        queue, d_W1, /*blocking*/ CL_TRUE, /*offset*/ 0,
        mem_size_W1, h_W1, 0, NULL, NULL));

  CALL_CL_SAFE(clEnqueueWriteBuffer(
        queue, d_W2, /*blocking*/ CL_TRUE, /*offset*/ 0,
        mem_size_W2, h_W2, 0, NULL, NULL));

  CALL_CL_SAFE(clEnqueueWriteBuffer(
        queue, d_y, /*blocking*/ CL_TRUE, /*offset*/ 0,
        mem_size_y, h_y, 0, NULL, NULL));

  cl_int wX = WX;
  cl_int wz2 = Wz2;
  cl_int wa2 = Wa2;
  cl_int ha2 = Ha2;
  cl_int hX = HX;

  size_t local_size[] = { 1, 1 };
  size_t global_size[] = {n, k };


  timestamp_type time1, time2;
  double elapsed;
  int h;
  get_timestamp(&time1);
  for(h = 0; h < 100; h++){
    local_size[0] = 4;
    local_size[1] = 4;
    global_size[0] = ceil(n/local_size[0]) * local_size[0];
    global_size[1] = ceil(k/local_size[1]) * local_size[1];

//    global_size[0] = n;
//    global_size[1] = k;   


    CALL_CL_SAFE(clSetKernelArg(knl1, 0, sizeof(d_X), &d_X));
    CALL_CL_SAFE(clSetKernelArg(knl1, 1, sizeof(d_W1), &d_W1));
    CALL_CL_SAFE(clSetKernelArg(knl1, 2, sizeof(d_z2), &d_z2));
    CALL_CL_SAFE(clSetKernelArg(knl1, 3, sizeof(d_a2), &d_a2));
    CALL_CL_SAFE(clSetKernelArg(knl1, 4, sizeof(wz2), &wz2));
    CALL_CL_SAFE(clSetKernelArg(knl1, 5, sizeof(wX), &wX));
    CALL_CL_SAFE(clSetKernelArg(knl1, 6, sizeof(hX), &hX));


    CALL_CL_SAFE(clEnqueueNDRangeKernel(queue, knl1, 2, NULL,
          global_size, local_size, 0, NULL, NULL));


/*    CALL_CL_SAFE(clEnqueueReadBuffer(
          queue, d_z2,  CL_TRUE,  0,
          mem_size_z2, h_z2,
          0, NULL, NULL));

printf("z2 is %f\n", h_z2[0]);*/

    local_size[0] = 1;
    local_size[1] = 1;
    global_size[0] = n;
    global_size[1] = 1;

    CALL_CL_SAFE(clSetKernelArg(knl2, 0, sizeof(d_a2), &d_a2));
    CALL_CL_SAFE(clSetKernelArg(knl2, 1, sizeof(d_W2), &d_W2));
    CALL_CL_SAFE(clSetKernelArg(knl2, 2, sizeof(d_z3), &d_z3));
    CALL_CL_SAFE(clSetKernelArg(knl2, 3, sizeof(d_yHat), &d_yHat));
    CALL_CL_SAFE(clSetKernelArg(knl2, 4, sizeof(wa2), &wa2));


    CALL_CL_SAFE(clEnqueueNDRangeKernel(queue, knl2, 2, NULL,
          global_size, local_size, 0, NULL, NULL));
 

    CALL_CL_SAFE(clEnqueueReadBuffer(
          queue, d_yHat,  CL_TRUE,  0,
          mem_size_yHat, h_yHat,
          0, NULL, NULL));


  // compute cost
    float J = 0;
    for (i = 0; i < n; i++){
      J += (h_y[i]- h_yHat[i]) * (h_y[i]- h_yHat[i]);
    }
    printf("Cost is %f\n", 0.5*J);


    global_size[0] = n;
    global_size[1] = 1;

    CALL_CL_SAFE(clSetKernelArg(knl3, 0, sizeof(d_yHat), &d_yHat));
    CALL_CL_SAFE(clSetKernelArg(knl3, 1, sizeof(d_y), &d_y));
    CALL_CL_SAFE(clSetKernelArg(knl3, 2, sizeof(d_z3), &d_z3));
    CALL_CL_SAFE(clSetKernelArg(knl3, 3, sizeof(d_delta3), &d_delta3));

    CALL_CL_SAFE(clEnqueueNDRangeKernel(queue, knl3, 2, NULL,
          global_size, local_size, 0, NULL, NULL));




    global_size[0] = k;
    global_size[1] = 1;

    CALL_CL_SAFE(clSetKernelArg(knl7, 0, sizeof(d_a2), &d_a2));
    CALL_CL_SAFE(clSetKernelArg(knl7, 1, sizeof(d_delta3), &d_delta3));
    CALL_CL_SAFE(clSetKernelArg(knl7, 2, sizeof(d_dJdW2), &d_dJdW2));
    CALL_CL_SAFE(clSetKernelArg(knl7, 3, sizeof(ha2), &ha2));
    CALL_CL_SAFE(clSetKernelArg(knl7, 4, sizeof(wa2), &wa2));


    CALL_CL_SAFE(clEnqueueNDRangeKernel(queue, knl7, 2, NULL,
          global_size, local_size, 0, NULL, NULL));



    global_size[0] = n;
    global_size[1] = k;

    CALL_CL_SAFE(clSetKernelArg(knl4, 0, sizeof(d_delta3), &d_delta3));
    CALL_CL_SAFE(clSetKernelArg(knl4, 1, sizeof(d_W2), &d_W2));
    CALL_CL_SAFE(clSetKernelArg(knl4, 2, sizeof(d_z2), &d_z2));
    CALL_CL_SAFE(clSetKernelArg(knl4, 3, sizeof(d_delta2), &d_delta2));
    CALL_CL_SAFE(clSetKernelArg(knl4, 4, sizeof(ha2), &ha2));
    CALL_CL_SAFE(clSetKernelArg(knl4, 5, sizeof(wa2), &wa2));



    CALL_CL_SAFE(clEnqueueNDRangeKernel(queue, knl4, 2, NULL,
          global_size, local_size, 0, NULL, NULL));


    local_size[0] = 4;
    local_size[1] = 4;
    global_size[0] = ceil(m/local_size[0]) * local_size[0];
    global_size[1] = ceil(k/local_size[1]) * local_size[1];

//    global_size[0] = m;
//    global_size[1] = k;

    CALL_CL_SAFE(clSetKernelArg(knl5, 0, sizeof(d_X), &d_X));
    CALL_CL_SAFE(clSetKernelArg(knl5, 1, sizeof(d_delta2), &d_delta2));
    CALL_CL_SAFE(clSetKernelArg(knl5, 2, sizeof(d_dJdW1), &d_dJdW1));
    CALL_CL_SAFE(clSetKernelArg(knl5, 3, sizeof(wX), &wX));
    CALL_CL_SAFE(clSetKernelArg(knl5, 4, sizeof(hX), &hX));
    CALL_CL_SAFE(clSetKernelArg(knl5, 5, sizeof(wz2), &wz2));



    CALL_CL_SAFE(clEnqueueNDRangeKernel(queue, knl5, 2, NULL,
          global_size, local_size, 0, NULL, NULL));


    CALL_CL_SAFE(clSetKernelArg(knl6, 0, sizeof(d_dJdW1), &d_dJdW1));
    CALL_CL_SAFE(clSetKernelArg(knl6, 1, sizeof(d_W1), &d_W1));
    CALL_CL_SAFE(clSetKernelArg(knl6, 2, sizeof(WW1), &WW1));

    CALL_CL_SAFE(clFinish(queue));
    CALL_CL_SAFE(clEnqueueNDRangeKernel(queue, knl6, 2, NULL,
            global_size, local_size, 0, NULL, NULL));

    CALL_CL_SAFE(clFinish(queue));

    local_size[0] = 1;
    local_size[1] = 1;
    global_size[0] = k;
    global_size[1] = 1;

    CALL_CL_SAFE(clSetKernelArg(knl6, 0, sizeof(d_dJdW2), &d_dJdW2));
    CALL_CL_SAFE(clSetKernelArg(knl6, 1, sizeof(d_W2), &d_W2));
    CALL_CL_SAFE(clSetKernelArg(knl6, 2, sizeof(WW2), &WW2));

    CALL_CL_SAFE(clEnqueueNDRangeKernel(queue, knl6, 2, NULL,
            global_size, local_size, 0, NULL, NULL));


  }
  get_timestamp(&time2);
  elapsed = timestamp_diff_in_seconds(time1,time2);
  printf("Time consumed : %f\n", elapsed);   

  CALL_CL_SAFE(clReleaseMemObject(d_X));
  CALL_CL_SAFE(clReleaseMemObject(d_W1));
  CALL_CL_SAFE(clReleaseMemObject(d_z2));
  CALL_CL_SAFE(clReleaseMemObject(d_a2));
  CALL_CL_SAFE(clReleaseKernel(knl1));
  CALL_CL_SAFE(clReleaseCommandQueue(queue));
  CALL_CL_SAFE(clReleaseContext(ctx));
}
//gcc -o NN n_host.c cl-helper.c helper.c -framework OpenCL -lm
