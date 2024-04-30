#include <inttypes.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <string.h>

double t_serial = 0.0;
double t_parallel = 0.0;
struct particle
{
    float x, y, z;
};
double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}
const float G = 6.67e-11;
void calculate_forces(struct particle *p, struct particle *f[], float *m, int n, int t)
{
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    for (int i = 0; i < n; i++)
    {
        f[tid][i].x = 0;
        f[tid][i].y = 0;
        f[tid][i].z = 0;
    }
#pragma omp for schedule(dynamic, 8)
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                float dist = sqrtf(powf(p[i].x - p[j].x, 2) + powf(p[i].y - p[j].y, 2) +
                                   powf(p[i].z - p[j].z, 2));
                float mag = (G * m[i] * m[j]) / powf(dist, 2);
                struct particle dir = {.x = p[j].x - p[i].x, .y = p[j].y - p[i].y, .z = p[j].z - p[i].z};
                f[tid][i].x += mag * dir.x / dist;
                f[tid][i].y += mag * dir.y / dist;
                f[tid][i].z += mag * dir.z / dist;
                f[tid][j].x -= mag * dir.x / dist;
                f[tid][j].y -= mag * dir.y / dist;
                f[tid][j].z -= mag * dir.z / dist;
            }
        }          // barrier
#pragma omp single // Итоговый вектор сил сформируем в первой строке – f[0][i]
        {
            for (int i = 0; i < n; i++)
            {
                for (int tid = 1; tid < nthreads; tid++)
                {
                    f[0][i].x += f[tid][i].x;
                    f[0][i].y += f[tid][i].y;
                    f[0][i].z += f[tid][i].z;
                }
            }
        }
} // calculate_forces
void move_particles(struct particle *p, struct particle *f[], struct particle *v, float *m, int n,
                    double dt, int t)
{
#pragma omp for
        for (int i = 0; i < n; i++)
        {
            struct particle dv = {
                .x = f[0][i].x / m[i] * dt,
                .y = f[0][i].y / m[i] * dt,
                .z = f[0][i].z / m[i] * dt,
            };
            struct particle dp = {
                .x = (v[i].x + dv.x / 2) * dt,
                .y = (v[i].y + dv.y / 2) * dt,
                .z = (v[i].z + dv.z / 2) * dt,
            };
            v[i].x += dv.x;
            v[i].y += dv.y;
            v[i].z += dv.z;
            p[i].x += dp.x;
            p[i].y += dp.y;
            p[i].z += dp.z;
            // f[i].x = f[i].y = f[i].z = 0;
        }
}

void calculate_forces_serial(struct particle *p, struct particle *f, float *m, int n)
{
    for (int i = 0; i < n - 1; i++)
    {
        for (int j = i + 1; j < n; j++)
        {
            // Вычисление силы, действующей на тело i со стороны j
            float dist = sqrtf(powf(p[i].x - p[j].x, 2) + powf(p[i].y - p[j].y, 2) +
                               powf(p[i].z - p[j].z, 2));
            float mag = (G * m[i] * m[j]) / powf(dist, 2);
            struct particle dir = {
                .x = p[j].x - p[i].x,
                .y = p[j].y - p[i].y,
                .z = p[j].z - p[i].z};
            // Сумма сил, действующих на тело i
            f[i].x += mag * dir.x / dist;
            f[i].y += mag * dir.y / dist;
            f[i].z += mag * dir.z / dist;
            // Сумма сил, действующих на тело j (симметричность)
            f[j].x -= mag * dir.x / dist;
            f[j].y -= mag * dir.y / dist;
            f[j].z -= mag * dir.z / dist;
        }
    }
}
void move_particles_serial(struct particle *p, struct particle *f, struct particle *v, float *m, int n,
                           double dt)
{
    for (int i = 0; i < n; i++)
    {
        struct particle dv = {
            .x = f[i].x / m[i] * dt,
            .y = f[i].y / m[i] * dt,
            .z = f[i].z / m[i] * dt,
        };
        struct particle dp = {
            .x = (v[i].x + dv.x / 2) * dt,
            .y = (v[i].y + dv.y / 2) * dt,
            .z = (v[i].z + dv.z / 2) * dt,
        };
        v[i].x += dv.x;
        v[i].y += dv.y;
        v[i].z += dv.z;
        p[i].x += dp.x;
        p[i].y += dp.y;
        p[i].z += dp.z;
        f[i].x = f[i].y = f[i].z = 0;
    }
}
void serial(int argc, char *argv[])
{
    double ttotal, tinit = 0, tforces = 0, tmove = 0;
    ttotal = wtime();
    int n = (argc > 1) ? atoi(argv[1]) : 10;
    char *filename = (argc > 2) ? argv[2] : NULL;
    tinit = -wtime();
    struct particle *p = malloc(sizeof(*p) * n); // Положение частиц (x, y, z)
    struct particle *f = malloc(sizeof(*f) * n); // Сила, действующая на каждую частицу (x, y, z)
    struct particle *v = malloc(sizeof(*v) * n); // Скорость частицы (x, y, z)
    float *m = malloc(sizeof(*m) * n);           // Масса частицы
    for (int i = 0; i < n; i++)
    {
        p[i].x = rand() / (float)RAND_MAX - 0.5;
        p[i].y = rand() / (float)RAND_MAX - 0.5;
        p[i].z = rand() / (float)RAND_MAX - 0.5;
        v[i].x = rand() / (float)RAND_MAX - 0.5;
        v[i].y = rand() / (float)RAND_MAX - 0.5;
        v[i].z = rand() / (float)RAND_MAX - 0.5;
        m[i] = rand() / (float)RAND_MAX * 10 + 0.01;
        f[i].x = f[i].y = f[i].z = 0;
    }
    tinit += wtime();
    double dt = 1e-5;
    for (double t = 0; t <= 1; t += dt)
    { // Цикл по времени (модельному)
        tforces -= wtime();
        calculate_forces_serial(p, f, m, n); // Вычисление сил – O(N^2)
        tforces += wtime();
        tmove -= wtime();
        move_particles_serial(p, f, v, m, n, dt); // Перемещение тел O(N)
        tmove += wtime();
    }
    t_serial = wtime() - ttotal;
    free(m);
    free(v);
    free(f);
    free(p);
}

int main(int argc, char *argv[])
{
    double ttotal, tinit = 0, tforces = 0, tmove = 0;
    ttotal = wtime();
    int n = (argc > 1) ? atoi(argv[1]) : 10;
    char *filename = (argc > 2) ? argv[2] : NULL;
    tinit = -wtime();
    struct particle *p = malloc(sizeof(*p) * n); // Положение частиц (x, y, z)
    struct particle *v = malloc(sizeof(*v) * n); // Скорость частицы (x, y, z)
    float *m = malloc(sizeof(*m) * n);           // Масса частицы
    // printf("\n------%d--------\n", omp_get_max_threads());
    struct particle *f[omp_get_max_threads()];
    for (int i = 0; i < omp_get_max_threads(); i++)
        f[i] = malloc(sizeof(struct particle) * n);
    for (int i = 0; i < n; i++)
    {
        p[i].x = rand() / (float)RAND_MAX - 0.5;
        p[i].y = rand() / (float)RAND_MAX - 0.5;
        p[i].z = rand() / (float)RAND_MAX - 0.5;
        v[i].x = rand() / (float)RAND_MAX - 0.5;
        v[i].y = rand() / (float)RAND_MAX - 0.5;
        v[i].z = rand() / (float)RAND_MAX - 0.5;
        m[i] = rand() / (float)RAND_MAX * 10 + 0.01;
        for (int j = 0; j < omp_get_max_threads(); j++)
            f[j][i].x = f[j][i].y = f[j][i].z = 0;
    }
    tinit += wtime();
    double dt = 1e-5;
    char buff[100] = "# Threads   Speedup\n";
    for (int t = 2; t <= 8; t += 2)
    {
        ttotal = wtime();
#pragma omp parallel num_threads(t) // Параллельный регион активируется один раз
        {
            for (double t = 0; t <= 1; t += dt)
            { // Цикл по времени (модельному)
                tforces -= wtime();
                calculate_forces(p, f, m, n, t); // Вычисление сил – O(N^2)
                tforces += wtime();
                tmove -= wtime();
                move_particles(p, f, v, m, n, dt, t); // Перемещение тел O(N)
                tmove += wtime();
            }
        }
        ttotal = wtime() - ttotal;
        printf("Threads: %d\n", t);
        printf("# NBody (n=%d)\n", n);
        printf("# Elapsed time (sec): ttotal %.6f, tinit %.6f, tforces %.6f, tmove %.6f\n",
               ttotal, tinit, tforces, tmove);
        char tmp[20];
        serial(argc, argv);
        sprintf(tmp, "%d\t\t%f\n", t, t_serial / ttotal);
        strcat(buff, tmp);
    }
    strcat(buff, "\0");
    if (filename)
    {
        FILE *fout = fopen(filename, "w");
        if (!fout)
        {
            fprintf(stderr, "Can't save file\n");
            exit(EXIT_FAILURE);
        }
        fwrite(buff, sizeof(char), strlen(buff), fout);
        fclose(fout);
    }
    free(m);
    free(v);
    for (int i = 0; i < omp_get_max_threads(); i++)
        free(f[i]);
    free(p);
    return 0;
}