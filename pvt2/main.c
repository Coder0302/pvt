#include <inttypes.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

int m = 15000, n = 15000;

void *xmalloc(size_t size) {
  void *ptr = malloc(size);
  if (ptr == NULL) {
    fprintf(stderr, "Failed to allocate memory\n");
    exit(EXIT_FAILURE);
  }
  return ptr;
}

double wtime() {
  struct timeval t;
  gettimeofday(&t, NULL);
  return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

void matrix_vector_product(double *a, double *b, double *c, int m, int n) {
  for (int i = 0; i < m; i++) {
    c[i] = 0.0;
    for (int j = 0; j < n; j++) c[i] += a[i * n + j] * b[j];
  }
}

/* matrix_vector_product_omp: Compute matrix-vector product c[m] = a[m][n] *
 * b[n] */
void matrix_vector_product_omp(double *a, double *b, double *c, int m, int n, int th) {
#pragma omp parallel num_threads(th)
  {
    int nthreads = omp_get_num_threads();
    int threadid = omp_get_thread_num();
    int items_per_thread = m / nthreads;
    int lb = threadid * items_per_thread;
    int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);
    for (int i = lb; i <= ub; i++) {
      c[i] = 0.0;
      for (int j = 0; j < n; j++) c[i] += a[i * n + j] * b[j];
    }
  }
}

double run_serial() {
  double *a, *b, *c;
  a = xmalloc(sizeof(*a) * m * n);
  b = xmalloc(sizeof(*b) * n);
  c = xmalloc(sizeof(*c) * m);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) a[i * n + j] = i + j;
  }
  for (int j = 0; j < n; j++) b[j] = j;
  double t = wtime();
  matrix_vector_product(a, b, c, m, n);
  t = wtime() - t;
  printf("Elapsed time (serial): %.6f sec.\n", t);
  free(a);
  free(b);
  free(c);
  return t;
}

double run_parallel(int th) {
  double *a, *b, *c;
  // Allocate memory for 2-d array a[m, n]
  a = xmalloc(sizeof(*a) * m * n);
  b = xmalloc(sizeof(*b) * n);
  c = xmalloc(sizeof(*c) * m);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) a[i * n + j] = i + j;
  }
  for (int j = 0; j < n; j++) b[j] = j;
  double t = wtime();
  matrix_vector_product_omp(a, b, c, m, n, th);
  t = wtime() - t;
  printf("Elapsed time (parallel): %.6f sec.\n", t);
  free(a);
  free(b);
  free(c);
  return t;
}

int main(int argc, char **argv) {
  printf("Matrix-vector product (c[m] = a[m, n] * b[n]; m = %d, n = %d)\n", m,
         n);
  printf("Memory used: %" PRIu64 " MiB\n",
         ((m * n + m + n) * sizeof(double)) >> 20);

  
  for (int j = 15000; j <= 25000; j+=5000)
  {
    for (int i = 2; i <= 8; i += 2)
    {
      n = j; m = j;
      printf("%d --> %d threads\n", n, i);
      printf("C: %.6f\n", run_serial()/run_parallel(i));
    }
  }
  return 0;
}