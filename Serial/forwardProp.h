#ifndef FOWRARDPROP_H_
#define FOWRARDPROP_H_

float sigmoidPrime(float x);

float sigmoid(float x);

void multiply(float *result, float *x, float *y, int n);

void arrayDot(float** result, float *x, float *y, int n, int k);

void transDot(float** result, float **x, float **y, int n, int m, int k);

void arrayTranDot(float* result, float **x, float *y, int n, int k);

void printMatrix(float **x, int n, int m);

void forward1(float **z2, float **x, float **w, int n, int m, int k);

void sigForward1(float **a2, float **x, int n, int k);

void forward2(float *result, float **x, float *y, int n, int k);

void sigForward2(float *result, float *x, int n);


#endif 