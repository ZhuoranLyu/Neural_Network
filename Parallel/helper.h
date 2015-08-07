#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>

#ifndef HELPER_H_
#define HELPER_H_

// Allocates a matrix with random float entries.
void randomMemInit(float* data, int size);
long LoadOpenCLKernel(char const* path, char **buf, bool add_nul);
void randomMemInit1(float* data, int size);

#endif 