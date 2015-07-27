#include "forwardProp.h"


float sigmoidPrime(float x){
	
	float y = exp(-x)/((1 + exp(-x)) * (1 + exp(-x)));
	return y;
}

float sigmoid(float x){
	float y = 1 / (1 + exp(-x));
	return y;
}
/*
float sigmoidPrime(float x){
	float y = 1 - (tanh(x) * tanh(x));
	return y;
}

float sigmoid(float x){
	float y = tanh(x);
	return y;
}
/* Multiplication of two n*m matrices */
void multiply(float *result, float *x, float *y, int n){
	int i;
	for(i = 0; i < n; i++){
		result[i] = x[i] * y[i];
	}
}

/* x is of n * 1, y is of 1 * k */
void arrayDot(float** result, float *x, float *y, int n, int k){
	int i, j;
	for(i = 0; i < n; i++){
		for(j = 0; j < k; j++){
			result[i][j] = x[i] * y[j];
		}
	}
}

/* x is of n * m, y is of m * k 
float **dot(float **x, float **y, float** result, int n, int m, int k){
	int i, j, a;
	float **result = calloc(n, sizeof(float*));
	for(i = 0; i < n; i++){
		result[i] =calloc(k, sizeof(float));
	}
	for(i = 0; i < n; i++){
		for(j = 0; j < k; j++){

			for(a = 0; a < m; a++){
				result[i][j] += x[i][a] * y[a][j];
			}

		}
	}
	return result;
}
/* x is of m*n, y is of m*k, result should be n*k */
void transDot(float** result, float **x, float **y, int n, int m, int k){
	int i;
	int j;
	int a;
	/*
	for (i = 0; i < n; i++){
		for (j = 0; j < k; j++){
			printf("%f, ", x[i][j]);
		}
		printf("\n");
	}
	*/

	for(i = 0; i < n; i++){
		for(j = 0; j < k; j++){
			for(a = 0; a < m; a++){
				result[i][j] += x[a][i] * y[a][j];
			}
		}
	}
}

 /*x is of n*k, y is of n*1 */
void arrayTranDot(float* result, float **x, float *y, int n, int k){
	int i, j;
	for(i = 0; i < k; i++){
		for(j = 0; j < n; j++){
			result[i] += x[j][i] * y[j];
		}
	}
}

void printMatrix(float **x, int n, int m){
	int i, j;
	for(i = 0; i < n; i++){
		for(j = 0; j < m; j ++){
			printf("  %f", x[i][j]);
		}
		printf("\n");
	}
}


// forward propagation

/*  x is of n*m;
	w is of (m) * k
*/
void forward1(float **z2, float **x, float **w, int n, int m, int k){
	int i, j, a;
	for(i = 0; i < n; i++){
		for(j = 0; j < k; j++){
			z2[i][j] = 0.;
			for(a = 0; a < m; a++){
				z2[i][j] += x[i][a] * w[a][j];
			}
		}
	}
	return;
}

/* x is of n*k */
void sigForward1(float **a2, float **x, int n, int k){
	int i, j;
	for(i = 0; i < n; i++){
		for(j = 0; j < k; j++){
			a2[i][j] = sigmoid(x[i][j]);
		}
	}
	return;
}

/*x is of n*k, y is of k*1 */
void forward2(float *result, float **x, float *y, int n, int k){
	int i, j;
	for(i = 0; i < n; i++){
		result[i] = 0.;
		for(j = 0; j < k; j++){
			result[i] += x[i][j] * y[j];
		}
	}
	return;
}

void sigForward2(float *result, float *x, int n){
	int i;
	for(i = 0; i < n; i++){
		result[i] = sigmoid(x[i]);
	}
	return;
}
