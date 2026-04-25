// =============================================================
//  tests/test_matrix.cpp
//  Test unitari — compilato separatamente da make test_unit
// =============================================================
#include "../src/Matrix.h"
#include "../src/ParallelMatrixMult.h"
#include <iostream>
#include <cmath>
#include <cassert>

static int passed = 0, failed = 0;

#define CHECK(cond, name) \
    do { \
        if (cond) { std::cout << "[PASS] " << name << "\n"; ++passed; } \
        else      { std::cerr << "[FAIL] " << name << "\n"; ++failed; } \
    } while(0)

// ─── 1. Costruzione e dimensioni ────────────────────────────────────────────
void test_construction() {
    Matrix M(4, 6);
    CHECK(M.getRows() == 4, "getRows()");
    CHECK(M.getCols() == 6, "getCols()");

    bool all_zero = true;
    for (size_t i = 0; i < 4; ++i)
        for (size_t j = 0; j < 6; ++j)
            if (M.at(i, j) != 0.0f) all_zero = false;
    CHECK(all_zero, "costruttore inizializza a zero");
}

// ─── 2. Accesso e zeros() ───────────────────────────────────────────────────
void test_access() {
    Matrix M(5, 5);
    M.at(2, 3) = 9.25f;
    CHECK(M.at(2, 3) == 9.25f, "scrittura/lettura singolo elemento");

    M.zeros();
    bool all_zero = true;
    for (size_t i = 0; i < 5; ++i)
        for (size_t j = 0; j < 5; ++j)
            if (M.at(i, j) != 0.0f) all_zero = false;
    CHECK(all_zero, "zeros() azzera tutto");
}

// ─── 3. randomInit: non-zero e determinismo ──────────────────────────────────
void test_random_init() {
    Matrix M(16, 16);
    M.randomInit(42);

    bool has_nonzero = false;
    for (size_t i = 0; i < 16; ++i)
        for (size_t j = 0; j < 16; ++j)
            if (M.at(i, j) != 0.0f) has_nonzero = true;
    CHECK(has_nonzero, "randomInit produce valori non-zero");

    Matrix M2(16, 16);
    M2.randomInit(42);
    CHECK(M.isEqual(M2, 1e-7f), "stesso seed → stesso risultato (determinismo)");

    Matrix M3(16, 16);
    M3.randomInit(43);
    CHECK(!M.isEqual(M3, 1e-7f), "seed diverso → risultato diverso");
}

// ─── 4. isEqual ─────────────────────────────────────────────────────────────
void test_is_equal() {
    Matrix A(8, 8), B(8, 8);
    A.randomInit(1); B.randomInit(1);
    CHECK(A.isEqual(B, 1e-7f), "isEqual: matrici identiche");

    B.at(3, 3) += 1.0f;
    CHECK(!A.isEqual(B, 1e-2f), "isEqual: matrici diverse (modifica elemento)");

    Matrix C(8, 9);
    C.randomInit(1);
    CHECK(!A.isEqual(C, 1e-7f), "isEqual: dimensioni diverse → false");
}

// ─── 5. I/O binario round-trip ───────────────────────────────────────────────
void test_file_io() {
    Matrix A(32, 32);
    A.randomInit(77);
    A.saveToFile("output/test_io.bin");

    Matrix B(1, 1);
    B.loadFromFile("output/test_io.bin");

    CHECK(B.getRows() == 32 && B.getCols() == 32, "loadFromFile: dimensioni");
    CHECK(A.isEqual(B, 1e-7f), "I/O binario: round-trip esatto");
}

// ─── 6. Move constructor ─────────────────────────────────────────────────────
void test_move() {
    Matrix A(8, 8);
    A.randomInit(7);
    float val = A.at(4, 5);

    Matrix B(std::move(A));
    CHECK(B.at(4, 5) == val, "move constructor: valore preservato");
    CHECK(B.getRows() == 8 && B.getCols() == 8, "move constructor: dimensioni ok");
}

// ─── 7. Correttezza moltiplicazione: identità ────────────────────────────────
void test_mult_identity() {
    const size_t N = 64;
    Matrix A(N, N), I(N, N), R(N, N);
    A.randomInit(33);

    // Costruisci matrice identità
    for (size_t i = 0; i < N; ++i) I.at(i, i) = 1.0f;

    matrixMultSequential(A, I, R);
    CHECK(A.isEqual(R, 1e-4f), "A * I == A (seriale)");

    Matrix R_par(N, N);
    matrixMultParallel(A, I, R_par, 4, "static", 8);
    CHECK(A.isEqual(R_par, 1e-4f), "A * I == A (parallelo shared)");

    Matrix R_priv(N, N);
    matrixMultParallelPrivate(A, I, R_priv, 4, "static", 8);
    CHECK(A.isEqual(R_priv, 1e-4f), "A * I == A (parallelo private)");
}

// ─── 8. Consistenza shared vs private ───────────────────────────────────────
void test_shared_vs_private() {
    const size_t N = 128;
    Matrix A(N, N), B(N, N), C_shared(N, N), C_private(N, N);
    A.randomInit(10); B.randomInit(20);

    matrixMultParallel       (A, B, C_shared,  4, "dynamic", 16);
    matrixMultParallelPrivate(A, B, C_private, 4, "dynamic", 16);
    CHECK(C_shared.isEqual(C_private, 1e-2f),
          "shared == private (stesso risultato numerico)");
}

// ─── 9. Consistenza tra scheduling diversi ──────────────────────────────────
void test_schedules_agree() {
    const size_t N = 128;
    Matrix A(N, N), B(N, N);
    Matrix C_sta(N, N), C_dyn(N, N), C_gui(N, N);
    A.randomInit(55); B.randomInit(66);

    matrixMultParallel(A, B, C_sta, 4, "static",  16);
    matrixMultParallel(A, B, C_dyn, 4, "dynamic", 16);
    matrixMultParallel(A, B, C_gui, 4, "guided",   1);

    CHECK(C_sta.isEqual(C_dyn, 1e-2f), "static == dynamic (scheduling)");
    CHECK(C_sta.isEqual(C_gui, 1e-2f), "static == guided  (scheduling)");
}

// ─── Entry point ─────────────────────────────────────────────────────────────
int main() {
    std::cout << "========================================\n";
    std::cout << " Test unitari – Matrix Multiplication\n";
    std::cout << "========================================\n\n";

    test_construction();
    test_access();
    test_random_init();
    test_is_equal();
    test_file_io();
    test_move();
    test_mult_identity();
    test_shared_vs_private();
    test_schedules_agree();

    std::cout << "\n----------------------------------------\n";
    std::cout << passed << " passed, " << failed << " failed\n";
    return (failed == 0) ? 0 : 1;
}
