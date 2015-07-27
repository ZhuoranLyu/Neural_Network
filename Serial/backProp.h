#ifndef BACKPROP_H_
#define BACKPROP_H_

float costFunction(float* yHat, float* y, int n);

void costFunctionPrime(float* delta3, float** delta2, float* temp, float** dJdW1,  float* dJdW2, float* yHat, float* y, float** z2, float* z3, float** a2, float** W1, float* W2, float** X,int n, int k, int m);

#endif