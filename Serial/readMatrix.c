#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//returns number of rows in matrix.
/* Note that C does not permit passing argument by reference.
   That's why we need the extra pointer, so that the value
   pointed to by the address is directly modified by the function.
 */
int* readMatrix(char* filename,double ***X){
  FILE *fp;
  fp = fopen(filename,"r");
  if (fp == NULL) {
    printf("ERROR: unable to read file.\n");
    return NULL;
  }
  char* line = NULL;
  size_t len = 0; //line length
  int lineLen = 0; //matrix length
  int linenum = 0; //matrix height

  //two passes, first pass to determine number of lines and line length
  // second pass to determine line length

  int passed = 0;
  while (getline(&line,&len,fp) != -1) {
    if (passed == 0) {
    char* elts = strtok(line," ,\t");
    while (elts != NULL) {
      lineLen++;
      elts = strtok(NULL," ,\t");
    }
    passed = 1;
    }
    linenum++;
  }
  fclose(fp);

  //open again for pass 2
  fp = fopen(filename,"r");
  *X = malloc(sizeof(double)*lineLen*linenum);
  int i,j;
  for (i = 0;i<linenum;i++) (*X)[i] = malloc(sizeof(double)*lineLen);
  for (i = 0;i<linenum;i++) {
    getline(&line,&len,fp);
    char* elts = strtok(line," ,\t");
    for (j=0;j<lineLen;j++) {
      (*X)[i][j] = strtod(elts,NULL);
      elts = strtok(NULL," ,\t");
    }
  }
  fclose(fp);
  int* dimpair = calloc(2,sizeof(int));
  dimpair[0] = lineLen; dimpair[1] = linenum;
  return dimpair;
}
