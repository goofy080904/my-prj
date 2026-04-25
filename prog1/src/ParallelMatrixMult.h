#ifndef PARALLELMATRIXMULT_H
#define PARALLELMATRIXMULT_H

#include "Matrix.h"
#include <string>
#include <vector>

// ─── Versione seriale di riferimento ────────────────────────────────────────
void matrixMultSequential(const Matrix& A, const Matrix& B, Matrix& C);

// ─── Versione parallela con chunk size fisso (per studio dello scheduling) ──
//     scheduleType: "static" | "dynamic" | "guided"
//     chunkSize: 0 = default OpenMP, >0 = chunk esplicito
void matrixMultParallel(const Matrix& A, const Matrix& B, Matrix& C,
                        int numThreads,
                        const std::string& scheduleType,
                        int chunkSize);

// ─── Versione parallela con variabili private esplicite ─────────────────────
//     Usa una variabile locale `sum` privata per thread invece di
//     accumulare direttamente su C: dimostra l'impatto di private vs shared.
void matrixMultParallelPrivate(const Matrix& A, const Matrix& B, Matrix& C,
                               int numThreads,
                               const std::string& scheduleType,
                               int chunkSize);

// ─── Benchmark completo: varia N, thread, schedule, chunk, private/shared ───
void benchmarkMatrixMult(const std::vector<int>& sizes,
                         const std::vector<int>& threadCounts,
                         const std::vector<std::string>& scheduleTypes,
                         const std::vector<int>& chunkSizes,
                         const std::string& outputFile);

#endif // PARALLELMATRIXMULT_H
