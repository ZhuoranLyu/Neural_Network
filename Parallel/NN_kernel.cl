/* a, b are arrays, c is the result
   mat_mult only uses the first dimensional work items
*/
#define TS 4

kernel void multiply(
      global float* a,
      global float* b,
      global float* c)
{
  int i;
  i = get_global_id(0);
  c[i] = a[i] * b[i];
  return;
}

/* a is n*1, y is 1*k, c should be n*k */
kernel void array_dot(
      global float* a,
      global float* b,
      global float* c,
      const int n,
      const int k)
{
  int i, j;
  i = get_global_id(0);
  j = get_global_id(1);
  if(i < n && j < k){
    c[i * k + j] = a[i] * b[j];
  }
  return;
}

/*a is m*n, b is m*k, c is n*k */
/*kernel void trans_dot(
      global float* a,
      global float* b,
      global float* c,
      const int n,
      const int m,
      const int k)
{
  int x, y, i;
  x = get_global_id(0);
  y = get_global_id(1);
  float val = 0.0;
  for(i = m-1; i >= 0; i--){
    val += a[i * n + x] * b[i * k + y];
  }
  c[x * k + y] = val;
  return;
}*/

kernel void trans_dot(
      global float* a,
      global float* b,
      global float* c,
      const int n,
      const int m,
      const int k)
{
  int row = get_local_id(0);
  int col = get_local_id(1);
  int globalRow = TS * get_group_id(0) + row;
  int globalCol = TS * get_group_id(1) + col;

  __local Asub[TS][TS];
  __local Bsub[TS][TS];

  float temp = 0.0;
  int numTiles = (m - 1) / TS + 1;
  int i, j;
  for(i = 0; i < numTiles; i++){
    int tileRow = TS * i + row;
    int tileCol = TS * i + col;

    if(globalRow < n && tileRow < m){
      Asub[row][col] = a[tileRow * n + globalRow];
    }else{
      Asub[row][col] = 0.0;
    }


    if(globalCol < k && tileRow < m){
      Bsub[row][col] = b[tileRow * k + globalCol];
    }else{
      Bsub[row][col] = 0.0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(j = 0; j < TS; j++){
      temp += Asub[j][row] * Bsub[j][col];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(globalRow < n && globalCol < k){
    c[globalRow * k + globalCol] = temp;
  }
}

/*a is n*k, b is n*1, c is k*1 */
kernel void array_trans_dot(
      global float* a,
      global float* b,
      global float* c,
      const int n,
      const int k)
{
  int i, j;
  i = get_global_id(0);
  float val = 0.0;
  for(j = 0; j < n; j++){
    val += a[j * k + i] * b[j];
  }
  c[i] = val;
  return;
}

/*a is n*m, b is m*k, c is n*k, d is n*k */
/*kernel void forward1(
      global float* a,
      global float* b,
      global float* c,
      global float* d,
      const int k,
      const int m,
      const int n)
{
  int x, y, i;
  x = get_global_id(0);
  y = get_global_id(1);
  float val = 0.0;
  for(i = 0; i < m; i++){
    val += a[x * m + i] * b[i * k + y];
  }
  if(x == 0 && y == 0){
    c[x * k + y] = val;
  }else{
  c[x * k + y] = val;
  }
  d[x * k + y] = 1 / (1 + exp(-val));
  return;
}*/

kernel void forward1(
      global float* a,
      global float* b,
      global float* c,
      global float* d,
      const int k,
      const int m,
      const int n)
{
  int row = get_local_id(0);
  int col = get_local_id(1);
  int globalRow = TS * get_group_id(0) + row;
  int globalCol = TS * get_group_id(1) + col;

  __local float Asub[TS][TS];
  __local float Bsub[TS][TS];

  float temp = 0.0;
  int numTiles = (m - 1) / TS + 1;
  int i, j;
  for(i = 0; i < numTiles; i++){
    int tileRow = TS * i + row;
    int tileCol = TS * i + col;
    if(globalRow < n && tileCol < m){
      Asub[row][col] = a[globalRow * m + tileCol];
    }else{
      Asub[row][col] = 0.0;
    }
    if(globalCol < k && tileRow < m){
      Bsub[row][col] = b[tileRow * k + globalCol];
    }else{
      Bsub[row][col] = 0.0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    for(j = 0; j < TS; j++){
      temp += Asub[row][j] * Bsub[j][col];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if(globalRow < n && globalCol < k){
    c[globalRow * k + globalCol] = temp;
    d[globalRow * k + globalCol] = 1 / (1 + exp(-temp));
  }
  return;
}

/* a is n*k, y is k*1, c is n*1, d is n*1 */
kernel void forward2(
      global float* a,
      global float* b,
      global float* c,
      global float* d,
      const int k)
{
  int i, j;
  i = get_global_id(0);
  float val = 0.0;
  for(j = 0; j < k; j++){
    val += a[i * k + j] * b[j];
  }
  c[i] = val;
  d[i] = 1 / (1 + exp(-val));
  return;
}

/*a is yHat, b is y, c is z3, d is result delta3 */
kernel void back1(
      global float* a,
      global float* b,
      global float* c,
      global float* d)
{
  int i;
  i = get_global_id(0);
  float temp;
  temp = a[i] - b[i];
  c[i] = exp(-c[i])/((1 + exp(-c[i]))*(1 + exp(-c[i])));
  d[i] = temp * c[i];
  return;
}


/* a is a2(n*k), b is delta3(n*1), c is W[m](1*k), d is z2(n*k), e is delta2(n*k), f is dJdW2(k*1)
   I assume there are n*k work items
 */
kernel void back2(
      global float* b,
      global float* c,
      global float* d,
      global float* e,
      const int n,
      const int k)
{
  int i, j, z;
  i = get_global_id(0);
  j = get_global_id(1);

  if(i < n && j < k){
    e[i * k + j] = b[i] * c[j];
  }
  float temp = d[i * k + j];
  e[i * k + j] = e[i * k + j] * exp(-temp)/((1 + exp(-temp)) * (1 + exp(-temp)));
  return;
}

/*a is a2(n*k), b is delta3(n*1), c is dJdW2(k*1)*/
kernel void back3(
  global float* a,
  global float* b,
  global float* c,
  const int n,
  const int k)
{
  int i, j;
  i = get_global_id(0);
  float val = 0.0;
  for(j = 0; j < n; j++){
    val += a[j * k + i] * b[j];
  }
  c[i] = val;
  return;
}

/*a is m*k, b is m*k */
kernel void update(
      global float* a,
      global float* b,
      const int k)
{
  float step = 0.01;
  int i, j;
  i = get_global_id(0);
  j = get_global_id(1);
  float temp = b[i * k + j];
  b[i * k + j] = temp - step * a[i * k + j];
  return;
}
