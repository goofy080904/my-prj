#include "ParallelMatrixMult.h"
#include <omp.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
//  Helper: imposta schedule a runtime in base alla stringa
// ─────────────────────────────────────────────────────────────────────────────
static void applySchedule(const std::string& scheduleType, int chunkSize) {
    int chunk = (chunkSize > 0) ? chunkSize : 1;
    if      (scheduleType == "dynamic") omp_set_schedule(omp_sched_dynamic, chunk);
    else if (scheduleType == "guided")  omp_set_schedule(omp_sched_guided,  chunk);
    else                                omp_set_schedule(omp_sched_static,  chunk);
}

// ─────────────────────────────────────────────────────────────────────────────
//  SERIALE: loop ikj (cache-friendly: B.at(k,j) è sequenziale in memoria)
// ─────────────────────────────────────────────────────────────────────────────
void matrixMultSequential(const Matrix& A, const Matrix& B, Matrix& C) {
    const size_t n = A.getRows();
    C.zeros();
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < n; ++k) {
            const float aik = A.at(i, k);   // valore scalare letto una volta
            for (size_t j = 0; j < n; ++j) {
                C.at(i, j) += aik * B.at(k, j);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  PARALLELA – variabili SHARED (versione "base")
//
//  Variabili SHARED:  A, B, C, n
//    - A e B sono in sola lettura → nessuna race condition
//    - C: ogni thread scrive su righe diverse (i è la variabile di loop
//      parallelizzata) → nessuna race condition
//    - n: costante di sola lettura
//
//  Variabili PRIVATE (implicite perché locali al corpo del loop):
//    - aik: ogni thread ha la propria copia sullo stack
//    - k, j: variabili di loop interne → private automaticamente
//
//  default(none) obbliga il compilatore a rifiutare variabili non classificate
//  esplicitamente: questo aumenta la chiarezza e previene bug.
// ─────────────────────────────────────────────────────────────────────────────
void matrixMultParallel(const Matrix& A, const Matrix& B, Matrix& C,
                        int numThreads,
                        const std::string& scheduleType,
                        int chunkSize) {
    const size_t n = A.getRows();
    C.zeros();
    omp_set_num_threads(numThreads);
    applySchedule(scheduleType, chunkSize);

    #pragma omp parallel for schedule(runtime) default(none) \
        shared(A, B, C, n)
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < n; ++k) {
            const float aik = A.at(i, k);  // privata implicitamente (locale)
            for (size_t j = 0; j < n; ++j) {
                C.at(i, j) += aik * B.at(k, j);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  PARALLELA – variabili PRIVATE esplicite (versione "private")
//
//  Differenza rispetto a matrixMultParallel:
//    - Ogni thread calcola un'intera riga di C in un buffer locale `row_buf`
//      (privato, allocato sullo stack del thread).
//    - Al termine della riga, copia il buffer su C con un unico memcpy.
//
//  Vantaggio: durante l'accumulo dei prodotti, il thread lavora su memoria
//  privata (in cache L1), riducendo il traffico verso la memoria condivisa.
//  Lo svantaggio è l'overhead del memcpy finale, che però è trascurabile.
//
//  Variabili SHARED:  A, B, C, n
//  Variabili PRIVATE: i, k, j, row_buf (dichiarati nel corpo → privati)
// ─────────────────────────────────────────────────────────────────────────────
void matrixMultParallelPrivate(const Matrix& A, const Matrix& B, Matrix& C,
                               int numThreads,
                               const std::string& scheduleType,
                               int chunkSize) {
    const size_t n = A.getRows();
    C.zeros();
    omp_set_num_threads(numThreads);
    applySchedule(scheduleType, chunkSize);

    #pragma omp parallel for schedule(runtime) default(none) \
        shared(A, B, C, n)
    for (size_t i = 0; i < n; ++i) {
        // row_buf è PRIVATO: vive sullo stack del thread, non su C (shared)
        // Ogni thread accumula su memoria locale → nessun falso sharing con C
        float row_buf[4096] = {};   // abbastanza grande per n ≤ 4096

        for (size_t k = 0; k < n; ++k) {
            const float aik = A.at(i, k);
            for (size_t j = 0; j < n; ++j) {
                row_buf[j] += aik * B.at(k, j);
            }
        }
        // Scrittura su C avviene una sola volta per riga (flush privato→shared)
        for (size_t j = 0; j < n; ++j)
            C.at(i, j) = row_buf[j];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  BENCHMARK
//  Per ogni combinazione (N, schedule, thread, chunk) misura:
//    - tempo seriale (baseline, misurato una volta per N)
//    - tempo parallelo versione "shared"
//    - tempo parallelo versione "private"
//    - speedup = ms_seq / ms_par
//  I/O da file escluso dalla misurazione (come richiesto dal testo).
// ─────────────────────────────────────────────────────────────────────────────
void benchmarkMatrixMult(const std::vector<int>& sizes,
                         const std::vector<int>& threadCounts,
                         const std::vector<std::string>& scheduleTypes,
                         const std::vector<int>& chunkSizes,
                         const std::string& outputFile) {

    std::ofstream csv(outputFile);
    if (!csv) throw std::runtime_error("Cannot open output: " + outputFile);

    // Intestazione CSV
    csv << "N,Schedule,Threads,ChunkSize,Variant,TimeMS,Speedup\n";

    for (int n : sizes) {
        std::cout << "\n=== N=" << n << " ===\n";

        Matrix A(n, n), B(n, n), C_seq(n, n);
        A.randomInit(42);
        B.randomInit(99);

        // ── Baseline seriale ──────────────────────────────────────────────
        auto t0 = std::chrono::high_resolution_clock::now();
        matrixMultSequential(A, B, C_seq);
        double ms_seq = std::chrono::duration<double, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();

        csv << n << ",sequential,1,0,sequential," << ms_seq << ",1.0\n";
        std::cout << "  sequential: " << ms_seq << " ms\n";

        // ── Loop su scheduling ────────────────────────────────────────────
        for (const auto& sched : scheduleTypes) {
            for (int chunk : chunkSizes) {
                for (int t : threadCounts) {

                    // --- Versione SHARED ---
                    {
                        Matrix C_par(n, n);
                        auto t1 = std::chrono::high_resolution_clock::now();
                        matrixMultParallel(A, B, C_par, t, sched, chunk);
                        double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - t1).count();
                        double speedup = ms_seq / ms;

                        csv << n << "," << sched << "," << t << "," << chunk
                            << ",shared," << ms << "," << speedup << "\n";
                        std::cout << "  shared   | " << sched
                                  << " chunk=" << chunk
                                  << " t=" << t
                                  << " -> " << ms << " ms (x" << speedup << ")\n";
                    }

                    // --- Versione PRIVATE ---
                    {
                        Matrix C_priv(n, n);
                        auto t1 = std::chrono::high_resolution_clock::now();
                        matrixMultParallelPrivate(A, B, C_priv, t, sched, chunk);
                        double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::high_resolution_clock::now() - t1).count();
                        double speedup = ms_seq / ms;

                        csv << n << "," << sched << "," << t << "," << chunk
                            << ",private," << ms << "," << speedup << "\n";
                        std::cout << "  private  | " << sched
                                  << " chunk=" << chunk
                                  << " t=" << t
                                  << " -> " << ms << " ms (x" << speedup << ")\n";
                    }
                }
            }
        }
    }

    std::cout << "\nBenchmark salvato in: " << outputFile << "\n";
}
