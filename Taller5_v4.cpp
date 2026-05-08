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

const int MAX_N = 12;                                                //este es un seguro contra cuelgues, es mejor dejarlo en 12, si se aument, sube exp el tiempo, esta ligado a tamanos

void generarMatriz(int n, int matrix[MAX_N][MAX_N]) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            matrix[i][j] = (i == j) ? 0 : (rand() % 10 + 1);
}

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

void backtrackingParaleloRec(int matrix[MAX_N][MAX_N], int n, int current, int end,
                             int dist, int *minDist, int visited[], int profundidad, int maxProfundidadParalela) {
    if (current == end) {
        #pragma omp critical
        {
            if (dist < *minDist) *minDist = dist;
        }
        return;
    }
    if (profundidad >= maxProfundidadParalela) {
        int localVisited[MAX_N];
        memcpy(localVisited, visited, sizeof(int) * n);
        backtrackingSecuencial(matrix, n, current, end, dist, minDist, visited);
        return;
    }
    for (int i = 0; i < n; i++) {
        if (matrix[current][i] != 0 && visited[i] == 0) {
            #pragma omp task firstprivate(i, dist)
            {
                int taskVisited[MAX_N];
                memcpy(taskVisited, visited, sizeof(int) * n);
                taskVisited[i] = 1;
                backtrackingParaleloRec(matrix, n, i, end, dist + matrix[current][i],
                                        minDist, taskVisited, profundidad + 1, maxProfundidadParalela);
            }
        }
    }
    #pragma omp taskwait
}

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
            backtrackingParaleloRec(matrix, n, start, end, 0, &minDist, visited, 0, maxProfundidadParalela);
        }
    }
    *resultado = minDist;
}

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

double medirParalelo(int matrix[MAX_N][MAX_N], int n, int start, int end, int nt, int *res, int maxProf) {
    int r;
    auto t1 = steady_clock::now();
    ejecutarParaleloTasks(matrix, n, start, end, nt, &r, maxProf);
    auto t2 = steady_clock::now();
    *res = r;
    return duration<double, milli>(t2 - t1).count();
}

double mediana(vector<double> &v) {
    sort(v.begin(), v.end());
    size_t sz = v.size();
    if (sz % 2 == 1) return v[sz / 2];
    return (v[sz / 2 - 1] + v[sz / 2]) / 2.0;
}

int main() {
    srand(static_cast<unsigned>(time(nullptr)));
    vector<int> tamanos = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};   // aca es el tamaño de las matrices    : esta ligado a MAX_N
    vector<int> nucleos = {1, 2, 3, 4};                          //aqui se va aumentando o disminullendo los nucleos
    int matrix[MAX_N][MAX_N];
    const int WARMUP = 2;
    const int REPS = 7;                                           // saca un promedio de ms que demora, mayor numero, mayor presicion

    // Encabezado
    cout << "\n" << string(85, '=') << endl;
    cout << left << setw(6)  << "N" 
         << setw(10) << "Modo" 
         << setw(10) << "Hilos" 
         << setw(20) << "Tiempo (ms)" 
         << setw(15) << "Speedup" 
         << "Resultado" << endl;
    cout << string(85, '-') << endl;

    for (int n : tamanos) {
        generarMatriz(n, matrix);
        int start = 0;
        int end = n - 1;
        int maxProf = 2;                                          // entre mayor el numero, mas ejecuciones en paralelo se ejecutara

        // Warmup para estabilizar caches
        int dummy;
        for (int w = 0; w < WARMUP; w++) medirSecuencial(matrix, n, start, end, &dummy);

        vector<double> tiemposSec;
        int resSec = 0;
        for (int r = 0; r < REPS; r++) {
            tiemposSec.push_back(medirSecuencial(matrix, n, start, end, &resSec));
        }
        double tiempoSecBase = mediana(tiemposSec);

        cout << left << setw(6)  << n 
             << setw(10) << "SEC" 
             << setw(10) << "1" 
             << fixed << setprecision(4) << setw(20) << tiempoSecBase 
             << setw(15) << "1.00x" 
             << resSec << endl;

        for (int nt : nucleos) {
            for (int w = 0; w < WARMUP; w++) medirParalelo(matrix, n, start, end, nt, &dummy, maxProf);

            vector<double> tiemposPar;
            int resPar = 0;
            for (int r = 0; r < REPS; r++) {
                tiemposPar.push_back(medirParalelo(matrix, n, start, end, nt, &resPar, maxProf));
            }
            double tiempoPar = mediana(tiemposPar);
            double speedup = tiempoSecBase / tiempoPar;

            cout << left << setw(6)  << n 
                 << setw(10) << "PAR" 
                 << setw(10) << nt 
                 << fixed << setprecision(4) << setw(20) << tiempoPar;

            string sUp = to_string(speedup).substr(0, to_string(speedup).find(".") + 3) + "x";
            cout << left << setw(15) << sUp
                 << resPar << endl;
        }
        cout << string(80, '-') << endl;
    }
    return 0;
}
