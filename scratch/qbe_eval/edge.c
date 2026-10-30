#include <stdio.h>
#include <math.h>
void gemm(int, const float *, const float *, float *);
void ref_gemm(int, const float *, const float *, float *);
void saxpy(int, float, const float *, float *);
void jacobi2d(int, const float *, float *);
void conv3x3(int, int, const float *, const float *, float *);
static float A[40*40], B[40*40], C1[40*40], C2[40*40], O1[40*40], O2[40*40];
int main(void) {
  int bad = 0;
  for (int i = 0; i < 1600; i++) { A[i] = (i * 7 % 13) * 0.1f; B[i] = (i * 5 % 11) * 0.2f; }
  for (int n = 0; n <= 21; n++) {
    for (int i = 0; i < 1600; i++) C1[i] = C2[i] = 0.5f;
    gemm(n, A, B, C1);
    for (int i = 0; i < n; i++) for (int k = 0; k < n; k++) for (int j = 0; j < n; j++) C2[i*n+j] += A[i*n+k] * B[k*n+j];
    for (int i = 0; i < 1600; i++) if (C1[i] != C2[i]) { bad++; printf("gemm n=%d idx=%d\n", n, i); break; }
    for (int i = 0; i < 1600; i++) O1[i] = O2[i] = 1.0f;
    saxpy(n, 0.5f, A, O1);
    for (int i = 0; i < n; i++) O2[i] = 0.5f * A[i] + O2[i];
    for (int i = 0; i < 1600; i++) if (O1[i] != O2[i]) { bad++; printf("saxpy n=%d\n", n); break; }
    for (int i = 0; i < 1600; i++) O1[i] = O2[i] = -1.0f;
    jacobi2d(n, A, O1);
    for (int i = 1; i < n - 1; i++) for (int j = 1; j < n - 1; j++)
      O2[i*n+j] = 0.2f * (A[i*n+j] + A[i*n+j-1] + A[i*n+j+1] + A[(i-1)*n+j] + A[(i+1)*n+j]);
    for (int i = 0; i < 1600; i++) if (O1[i] != O2[i]) { bad++; printf("jacobi n=%d idx=%d\n", n, i); break; }
    for (int w = 0; w <= 9; w++) {
      for (int i = 0; i < 1600; i++) O1[i] = O2[i] = -1.0f;
      conv3x3(n, w, A, B, O1);
      for (int y = 1; y < n - 1; y++) for (int x = 1; x < w - 1; x++) { float s = 0.0f;
        for (int dy = -1; dy <= 1; dy++) for (int dx = -1; dx <= 1; dx++) s += A[(y+dy)*w+x+dx] * B[(dy+1)*3+dx+1];
        O2[y*w+x] = s; }
      for (int i = 0; i < 1600; i++) if (O1[i] != O2[i]) { bad++; printf("conv h=%d w=%d idx=%d\n", n, w, i); break; }
    }
  }
  printf(bad ? "EDGE FAILURES: %d\n" : "edge cases ok (bit-exact)\n", bad);
  return bad != 0;
}
