#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <fcntl.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/cl.h>
#include <stdbool.h>
#include "util.h"
#include "forwardProp.h"
#include "backProp.h"

//#include "dataReader.h"

float max(float a, float b){
	return a > b ? a : b;
}

float min(float a, float b){
	return a < b ? a : b;
}

extern int* readMatrix(char* filename,double*** X);

int main(int argc, char **argv){

	int n; // example size
	int m; // input layer size
	int k = 50; // hidden layer size

	int i,j,p;

	float** X; // input matrix, n by m
	float* y; // output, n by 1
	
	srand(time(NULL));
        
	 //read the matrix from file
	char *filename = "qtrain.txt";
        int* dimpair = readmatrix(filename,&X);

	n = dimpair[1]; // example size
	m = dimpair[0] - 1;
	float* sum = calloc(m, sizeof(float));
	float* Max = calloc(m, sizeof(float));
	float* Min = calloc(m, sizeof(float));

	for (j = 0; j < m; j++){
		Max[j] = X[0][j];
		Min[j] = X[0][j];	
	}
	
	// Normalize the data
	for (i = 0; i < n; i++){
		for (j = 0; j < m; j++){
			sum[j] += X[i][j];	
			Max[j] = max(Max[j], X[i][j]);
			Min[j] = min(Min[j], X[i][j]);	
		}
	}

	for (i = 0; i < m; i++){
		sum[i] /= n;
	} 

	for (i = 0; i < n; i++){
		for (j = 0; j < m; j++){
			X[i][j] = (X[i][j] - Min[j]) / (Max[j] - Min[j]);
			X[i][j] -= 0.5;
	//		printf("%f,", X[i][j]);
		}
		y[i] /= 1000;
	//	printf("\n");
	}
	
	float J = 10.; // cost
	float** W1; // weight matrix, m by k
	float* W2; // weight matrix, k by 1
	float** z2; // n by k
	float** a2; // n by k
	float* z3; // n by 1
	float* yHat; // estimate output, n by 1

	float** dJdW1; // m by k
	float* dJdW2; // k by 1

	float threshold = 1;
	float step;

	float** deltaW1;
	float* deltaW2;
	// Init bias and weights for the network
	W1 = calloc(m, sizeof(float*)); // m by k
	W2 = calloc(k, sizeof(float));  // k by 1
	deltaW1 = calloc(m, sizeof(float*)); // m+1 by k
	deltaW2 = calloc(k, sizeof(float)); // m+1 by k
	
	for (i = 0; i < m; i++){
		W1[i] = calloc(k, sizeof(float));
		deltaW1[i] = calloc(k, sizeof(float));	
	}

	for (i = 0; i< m; i++){
		for (j = 0; j < k; j++){
			W1[i][j] = ((float)rand() / RAND_MAX);
			deltaW1[i][j] = 0.;
		}
	}

	for (j = 0; j < k; j++){
		W2[j] = ((float)rand() / RAND_MAX);
		deltaW2[j] = 0.;
	}
    z2 = calloc(n, sizeof(float*));
	for(i = 0; i < n; i++){
		z2[i] =calloc(k, sizeof(float));
	}

	a2 = calloc(n, sizeof(float *));
	for(i = 0; i < n; i++){
		a2[i] = calloc(k, sizeof(float));
	}

	z3 = calloc(n, sizeof(float));

	yHat = calloc(n, sizeof(float));

	dJdW1 = calloc(n, sizeof(float*));
	for(i = 0; i < n; i++){
		dJdW1[i] =calloc(k, sizeof(float));
	}

	dJdW2 = calloc(k, sizeof(float*));

	float* delta3 = calloc(n, sizeof(float)); // n by 1
	float** delta2 = calloc(n, sizeof(float *)); // n by k
	for(i = 0; i < n; i++){
		delta2[i] = calloc(k, sizeof(float));
	}
	float* temp = calloc(n, sizeof(float)); // store temp value for y

	step = 0.00001;
	timestamp_type time1, time2;
    get_timestamp(&time1);

	for (p = 0; p < 10; p++){
		forward1(z2, X, W1, n, m, k); // n by k

		sigForward1(a2, z2, n, k); // n by k

		forward2(z3, a2, W2, n, k); // n by 1

		sigForward2(yHat, z3, n); // n by 1
		
		J = costFunction(yHat, y, n);
		printf("cost is %f\n", J);
		costFunctionPrime(delta3, delta2, temp, dJdW1, dJdW2, yHat, y, z2, z3, a2, W1, W2, X, n, k, m);

		//printf("dJdW1\n");
		for (i = 0; i < m; i++){
			for (j = 0; j < k; j++){
				//printf("%f, ", W1[i][j]);
				//deltaW[i][j] = - step * dJdW[i][j];// + 0.9 * deltaW[i][j];
				W1[i][j] -= step * dJdW1[i][j];
			}
			//printf("\n");
		}

		for (j = 0; j < k; j++){
			//deltaW[i][j] = - step * dJdW[i][j];// + 0.9 * deltaW[i][j];
			W2[j] -= step * dJdW2[j];
		}

		//if (p % 10 == 0)
		printf("%dth iter.\n", p);
	}
	get_timestamp(&time2);

    double elapsed = timestamp_diff_in_seconds(time1,time2);

    printf("Time elapsed is %f seconds.\n", elapsed);

	J = costFunction(yHat, y, n);
	printf("%.15f\n", J);
	
	for (i = 0; i < n; i++){
		free(z2[i]);
	}
	free(z2);
	for (i = 0; i < n; i++){
		free(a2[i]);
	}
	free(a2);
	for (i = 0; i < m; i++){
		free(W1[i]);
		free(dJdW1[i]);	
	}
	free(W1);
	free(dJdW1);
	for (i = 0; i < n; i++){
		free(delta2[i]);
	}
	free(delta2);

	for (i = 0; i < n; i++){
		free(X[i]);
	}
	free(X);
	
	free(y); 
	free(z3);
	free(yHat);
	free(W2);
	free(dJdW2);
	free(delta3);
	free(temp);    
}
