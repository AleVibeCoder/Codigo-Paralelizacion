#include <iostream>
#include <vector>
#include <chrono>
#include <climits>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <omp.h>
#include <iomanip>
#include <sstream>

using namespace std;
using namespace std::chrono;

// Limite de seguridad: no subir de 12 ya que la complejidad crece factorial
const int MAX_N = 12;

// ---------------------------------------------------------------------------
// Genera matriz NxN con valores aleatorios 1-10, diagonal = 0
// ---------------------------------------------------------------------------
void generarMatriz(int n, int matrix[MAX_N][MAX_N]) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            matrix[i][j] = (i == j) ? 0 : (rand() % 10 + 1);
}

// ---------------------------------------------------------------------------
// SECUENCIAL: backtracking puro, 1 hilo, sirve como baseline
// ---------------------------------------------------------------------------
void backtrackingSecuencial(int matrix[MAX_N][MAX_N], int n, int current, int end,
                            int dist, int *minDist, int visited[]) {
    if (current == end) {
        if (dist < *minDist) *minDist = dist;
        return;
    }
    for (int i = 0; i < n; i++) {
        if (matrix[current][i] != 0 && visited[i] == 0) {
            visited[i] = 1;
            backtrackingSecuencial(matrix, n, i, end, dist + matrix[current][i], minDist, visited);
            visited[i] = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// PARALELO (v5):
//   - firstprivate(i, dist)     -> de v4: captura segura de variables por valor
//   - copia local taskVisited   -> de v4: evita race condition en visited
//   - SIN #pragma omp taskwait  -> correccion sobre v4: permite solapamiento real
//   - omp critical protege minDist (suficiente, no necesita taskwait)
//   - profundidad >= maxProfundidadParalela -> cae a secuencial para evitar
//     explosion de tareas en niveles profundos
// ---------------------------------------------------------------------------
void backtrackingParaleloRec(int matrix[MAX_N][MAX_N], int n, int current, int end,
                             int dist, int *minDist, int visited[],
                             int profundidad, int maxProfundidadParalela) {
    if (current == end) {
        #pragma omp critical
        {
            if (dist < *minDist) *minDist = dist;
        }
        return;
    }

    // Mas alla de la profundidad maxima, resolvemos secuencialmente
    // Usamos una copia local para no tocar el visited del padre (fix de v4)
    if (profundidad >= maxProfundidadParalela) {
        int localVisited[MAX_N];
        memcpy(localVisited, visited, sizeof(int) * n);
        backtrackingSecuencial(matrix, n, current, end, dist, minDist, localVisited);
        return;
    }

    for (int i = 0; i < n; i++) {
        if (matrix[current][i] != 0 && visited[i] == 0) {
            // firstprivate(i, dist): cada tarea tiene su propia copia de i y dist
            #pragma omp task firstprivate(i, dist)
            {
                int taskVisited[MAX_N];
                memcpy(taskVisited, visited, sizeof(int) * n);
                taskVisited[i] = 1;
                backtrackingParaleloRec(matrix, n, i, end,
                                        dist + matrix[current][i],
                                        minDist, taskVisited,
                                        profundidad + 1, maxProfundidadParalela);
            }
            // SIN taskwait aqui: las tareas corren en paralelo real
            // omp critical en el caso base garantiza la integridad de minDist
        }
    }
    // NO hay #pragma omp taskwait -> solapamiento de trabajo entre hilos
}

// ---------------------------------------------------------------------------
// Lanzador del modo paralelo con OpenMP Tasks
// ---------------------------------------------------------------------------
void ejecutarParaleloTasks(int matrix[MAX_N][MAX_N], int n, int start, int end,
                           int numThreads, int *resultado, int maxProfundidadParalela) {
    int visited[MAX_N];
    memset(visited, 0, sizeof(visited));
    visited[start] = 1;
    int minDist = INT_MAX;

    omp_set_num_threads(numThreads);
    #pragma omp parallel
    {
        #pragma omp single nowait
        {
            backtrackingParaleloRec(matrix, n, start, end, 0, &minDist,
                                    visited, 0, maxProfundidadParalela);
        }
    }
    *resultado = minDist;
}

// ---------------------------------------------------------------------------
// Medicion de tiempos
// ---------------------------------------------------------------------------
double medirSecuencial(int matrix[MAX_N][MAX_N], int n, int start, int end, int *res) {
    int visited[MAX_N];
    memset(visited, 0, sizeof(visited));
    visited[start] = 1;
    int minDist = INT_MAX;
    auto t1 = steady_clock::now();
    backtrackingSecuencial(matrix, n, start, end, 0, &minDist, visited);
    auto t2 = steady_clock::now();
    *res = minDist;
    return duration<double, milli>(t2 - t1).count();
}

double medirParalelo(int matrix[MAX_N][MAX_N], int n, int start, int end,
                     int nt, int *res, int maxProf) {
    int r;
    auto t1 = steady_clock::now();
    ejecutarParaleloTasks(matrix, n, start, end, nt, &r, maxProf);
    auto t2 = steady_clock::now();
    *res = r;
    return duration<double, milli>(t2 - t1).count();
}

// Mediana de un vector de doubles (reduce efecto de outliers del SO)
double mediana(vector<double> &v) {
    sort(v.begin(), v.end());
    size_t sz = v.size();
    if (sz % 2 == 1) return v[sz / 2];
    return (v[sz / 2 - 1] + v[sz / 2]) / 2.0;
}

// ---------------------------------------------------------------------------
// MAIN
// ---------------------------------------------------------------------------
int main() {
    srand(static_cast<unsigned>(time(nullptr)));

    // Tamaños de matriz: ligado a MAX_N
    vector<int> tamanos = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};

    // Cantidad de hilos a evaluar
    vector<int> nucleos = {1, 2, 3, 4};

    int matrix[MAX_N][MAX_N];
    const int WARMUP = 2;   // calentamiento para estabilizar caches
    const int REPS   = 7;   // repeticiones: mediana de 7 reduce ruido del SO

    // Profundidad maxima de paralelizacion:
    // 2 genera hasta n^2 tareas en los primeros niveles, buen balance overhead/speedup
    // Subir este valor aumenta paralelismo pero tambien overhead de tareas
    const int MAX_PROF = 2;

    // Encabezado de tabla
    cout << "\n" << string(85, '=') << "\n";
    cout << left
         << setw(6)  << "N"
         << setw(10) << "Modo"
         << setw(10) << "Hilos"
         << setw(20) << "Tiempo (ms)"
         << setw(15) << "Speedup"
         << "Resultado\n";
    cout << string(85, '-') << "\n";

    for (int n : tamanos) {
        generarMatriz(n, matrix);
        int start = 0;
        int end   = n - 1;

        // Warmup secuencial
        int dummy;
        for (int w = 0; w < WARMUP; w++)
            medirSecuencial(matrix, n, start, end, &dummy);

        // Medir secuencial
        vector<double> tiemposSec;
        int resSec = 0;
        for (int r = 0; r < REPS; r++)
            tiemposSec.push_back(medirSecuencial(matrix, n, start, end, &resSec));
        double tiempoSecBase = mediana(tiemposSec);

        cout << left
             << setw(6)  << n
             << setw(10) << "SEC"
             << setw(10) << "1"
             << fixed << setprecision(4) << setw(20) << tiempoSecBase
             << setw(15) << "1.00x"
             << resSec << "\n";

        // Medir paralelo para cada cantidad de hilos
        for (int nt : nucleos) {
            // Warmup paralelo
            for (int w = 0; w < WARMUP; w++)
                medirParalelo(matrix, n, start, end, nt, &dummy, MAX_PROF);

            vector<double> tiemposPar;
            int resPar = 0;
            for (int r = 0; r < REPS; r++)
                tiemposPar.push_back(medirParalelo(matrix, n, start, end, nt, &resPar, MAX_PROF));

            double tiempoPar = mediana(tiemposPar);
            double speedup   = tiempoSecBase / tiempoPar;

            // Formatear speedup como "X.XXx"
            ostringstream ssUp;
            ssUp << fixed << setprecision(2) << speedup << "x";

            cout << left
                 << setw(6)  << n
                 << setw(10) << "PAR"
                 << setw(10) << nt
                 << fixed << setprecision(4) << setw(20) << tiempoPar
                 << setw(15) << ssUp.str()
                 << resPar << "\n";
        }
        cout << string(80, '-') << "\n";
    }

    return 0;
}
