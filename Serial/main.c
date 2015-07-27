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


int main(int argc, char **argv){

	int n; // example size
	int m; // input layer size
	int k = 5; // hidden layer size

	int i,j,p;

	float** X; // input matrix, n by m
	float* y; // output, n by 1
	
	srand(time(NULL));
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
	X = calloc(lineNum, sizeof(float*));
	y = calloc(lineNum, sizeof(float));

	for (i = 0;i<lineNum;i++) {
		X[i] = calloc((lineLen-1), sizeof(float));
	}

	for (i = 0;i<lineNum;i++) {
		getline(&line,&len,fp);
		char* elts = strtok(line," ,\t");
		for (j=0;j<lineLen-1;j++) {
			X[i][j] = strtod(elts,NULL);
			elts = strtok(NULL," ,\t");
		}
		y[i] = strtod(elts,NULL);
		elts = strtok(NULL," ,\t");
		free(elts);
	}
	fclose(fp);

	n = lineNum; // example size
	m = lineLen - 1;
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
			//X[i][j] /= 100;
		}
		y[i] /= 100;
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
			W1[i][j] = 1.;//+ 1. / (1. + i * k + j);//((float)rand() / RAND_MAX);
			deltaW1[i][j] = 0.;
		}
	}

	for (j = 0; j < k; j++){
		W2[j] = 1.;// + 1./(1. + j);//((float)rand() / RAND_MAX);
		deltaW2[j] = 0.;
	}
    z2 = calloc(n, sizeof(float*));
	for(i = 0; i < n; i++){
		z2[i] =calloc(k, sizeof(float));
	}
	//z2[0] = 1.;
	a2 = calloc(n, sizeof(float *));
	for(i = 0; i < n; i++){
		a2[i] = calloc(k, sizeof(float));
	}

	z3 = calloc(n, sizeof(float));

	yHat = calloc(n, sizeof(float));

	dJdW1 = calloc(m, sizeof(float*));
	for(i = 0; i < m; i++){
		dJdW1[i] =calloc(k, sizeof(float));
	}

	dJdW2 = calloc(k, sizeof(float));

	float* delta3 = calloc(n, sizeof(float)); // n by 1
	float** delta2 = calloc(n, sizeof(float *)); // n by k
	for(i = 0; i < n; i++){
		delta2[i] = calloc(k, sizeof(float));
	}
	float* temp = calloc(n, sizeof(float)); // store temp value for y

	step = 0.1;
	timestamp_type time1, time2;
    get_timestamp(&time1);

	for (p = 0; p < 50; p++){
		float sum = 0;
		for (i = 0; i < n; i++){
			for (j = 0; j < m; j++){
				sum += X[i][j];
			}
		}
		printf("Sum of X is %.15f, ", sum);

		forward1(z2, X, W1, n, m, k); // n by k
		
		printf("%d\n", n);
		sum = 0;
		for (i = 0; i < m; i++){
			for (j = 0; j < k; j++){
				sum += W1[i][j];
				printf("%.15f, ", W1[i][j]);
			}
		}
		printf("Sum of W1 is \n");

		printf("%.15f, ", sum);
		
		printf("z2 is \n");
		for (i = 0; i < n; i++){
			for (j = 0; j < k; j++){
				printf("%.15f, ", z2[i][j]);
			}
		}
		/*
		sum = 0;
		for (i = 0; i < n; i++){
			for (j = 0; j < k; j++){
				sum += z2[i][j];
				printf("%.15f, ", z2[i][j]);
			}
		}
		printf("\nSum of z2 is \n");
		printf("%.15f, ", sum);
		*/
		sigForward1(a2, z2, n, k); // n by k
		/*
		printf("a2 is \n");
		for (j = 0; j < 50; j++){
			printf("%.15f, ", a2[0][j]);
		}
		*/

		forward2(z3, a2, W2, n, k); // n by 1
		/*
		printf("z3 is \n");
		for (j = 0; j < 10; j++){
			printf("%.15f, ", z3[j]);
		}
		*/
		sum = 0;
		printf("z3 is \n");
		for (j = 0; j < n; j++){
			sum += z3[j];
		}
		printf("%.15f, ", sum);
		sigForward2(yHat, z3, n); // n by 1

		J = costFunction(yHat, y, n);
		printf("cost is %f\n", J);
		costFunctionPrime(delta3, delta2, temp, dJdW1, dJdW2, yHat, y, z2, z3, a2, W1, W2, X, n, k, m);
		
		
		
		/*
		printf("delta3 is \n");
		for (j = 0; j < 50; j++){
			printf("%.15f, ",delta3[j]);
		}
		
		//printf("X is \n");
		for (j = 0; j < 50; j++){
			//printf("%f, ",X[0][j]);
		}

		/*
		printf("delta2 is \n");
		for (j = 0; j < 50; j++){
			printf("%.20f, ",delta2[0][j]);
		}
				printf("dJdW2 is \n");
		for (j = 0; j < 50; j++){
			printf("%.20f, ",dJdW2[j]);
		}
		*/

		//printf("dJdW1\n");
		for (i = 0; i < m; i++){
			for (j = 0; j < k; j++){
				//printf("%.15f, ", dJdW1[i][j]);
				//deltaW[i][j] = - step * dJdW[i][j];// + 0.9 * deltaW[i][j];
				W1[i][j] -= step * dJdW1[i][j];
			}
			//printf("\n");
		}


printf("\n");
		for (j = 0; j < k; j++){
			//deltaW[i][j] = - step * dJdW[i][j];// + 0.9 * deltaW[i][j];
			W2[j] -= step * dJdW2[j];
			//printf("%f, ", dJdW2[j]);
		}
		sum = 0;
		for (i = 0; i < k; i++){
			sum += W2[i];
		}
		printf("\nSum of W2 is \n");
		printf("%.15f, ", sum);
		//printf("%dth iter.\n", p);
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
