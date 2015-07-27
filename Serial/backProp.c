#include "forwardProp.h"
#include "backProp.h"

float costFunction(float* yHat, float* y, int n){
	float J = 0;
	int i = 0;
	for (i = 0; i < n; i++){
		J += (y[i]- yHat[i]) * (y[i]- yHat[i]);
	}
	return 0.5*J;
}

void costFunctionPrime(float* delta3, float** delta2, float* temp, float** dJdW1,  float* dJdW2, float* yHat, float* y, float** z2, float* z3, float** a2, float** W1, float* W2, float** X,int n, int k, int m){
	int i, j;
	for (i = 0; i < n; i++){
		temp[i] = - (y[i] - yHat[i]);
		z3[i] = sigmoidPrime(z3[i]);
	}


	multiply(delta3, temp, z3, n);

	arrayTranDot(dJdW2, a2, delta3, n, k);


	arrayDot(delta2, delta3, W2, n, k);

	for (i = 0; i < n; i++){
		for (j = 0; j < k; j++){
			delta2[i][j] *= sigmoidPrime(z2[i][j]);
		}
	}

	transDot(dJdW1, X, delta2, m, n, k);

	return;
}

