/* cimo: the OpenMP runtime functions cimo has (runtime/omp.c). Loops run on several cores with
   `#pragma omp parallel for`; see the documentation's "Parallel loops". */
#ifndef _OMP_H
#define _OMP_H

typedef struct { volatile int __cimo_word; } omp_lock_t;  /* 4 bytes, as gcc's */

int omp_get_thread_num(void);
int omp_get_num_threads(void);
int omp_get_max_threads(void);
void omp_set_num_threads(int);
int omp_get_num_procs(void);
int omp_in_parallel(void);
int omp_get_thread_limit(void);
int omp_get_level(void);
int omp_get_active_level(void);
int omp_get_dynamic(void);
void omp_set_dynamic(int);
int omp_get_nested(void);
void omp_set_nested(int);
int omp_get_max_active_levels(void);
void omp_set_max_active_levels(int);
double omp_get_wtime(void);
double omp_get_wtick(void);

void omp_init_lock(omp_lock_t *);
void omp_destroy_lock(omp_lock_t *);
void omp_set_lock(omp_lock_t *);
void omp_unset_lock(omp_lock_t *);
int omp_test_lock(omp_lock_t *);

#endif
