#include "Matrix.h"
#include "ParallelMatrixMult.h"
#include <iostream>
#include <string>
#include <vector>
#include <cstdlib>  // per system()

// ─────────────────────────────────────────────────────────────────────────────
//  Uso:
//    ./matmul test          → correttezza su matrice 256x256
//    ./matmul benchmark     → benchmark completo, risultati in output/results.csv
//                            + genera automaticamente i grafici
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Uso: ./matmul [test|benchmark]\n";
        return 1;
    }

    const std::string mode = argv[1];

    // ─── TEST DI CORRETTEZZA ─────────────────────────────────────────────────
    if (mode == "test") {
        const int N = 256;
        Matrix A(N, N), B(N, N), C_seq(N, N), C_par(N, N), C_priv(N, N);
        A.randomInit(42);
        B.randomInit(99);

        matrixMultSequential(A, B, C_seq);

        // Test versione shared
        matrixMultParallel(A, B, C_par, 4, "static", 16);
        if (C_seq.isEqual(C_par))
            std::cout << "[PASS] shared parallel == sequential\n";
        else
            std::cerr << "[FAIL] shared parallel != sequential\n";

        // Test versione private
        matrixMultParallelPrivate(A, B, C_priv, 4, "static", 16);
        if (C_seq.isEqual(C_priv))
            std::cout << "[PASS] private parallel == sequential\n";
        else
            std::cerr << "[FAIL] private parallel != sequential\n";

        // Test schedule dynamic
        Matrix C_dyn(N, N);
        matrixMultParallel(A, B, C_dyn, 4, "dynamic", 8);
        if (C_seq.isEqual(C_dyn))
            std::cout << "[PASS] dynamic schedule == sequential\n";
        else
            std::cerr << "[FAIL] dynamic schedule != sequential\n";

        // Test schedule guided
        Matrix C_gui(N, N);
        matrixMultParallel(A, B, C_gui, 4, "guided", 1);
        if (C_seq.isEqual(C_gui))
            std::cout << "[PASS] guided schedule == sequential\n";
        else
            std::cerr << "[FAIL] guided schedule != sequential\n";
    }

    // ─── BENCHMARK ───────────────────────────────────────────────────────────
    else if (mode == "benchmark") {
        // Dimensioni matrici: multipli del numero di core, potenze di 2
        const std::vector<int> sizes      = {512, 1024, 2048};
        // Numero di thread
        const std::vector<int> threads    = {1, 2, 4, 8};
        // Scheduling
        const std::vector<std::string> scheds = {"static", "dynamic", "guided"};
        // Chunk sizes: studio dell'impatto del granularity
        const std::vector<int> chunks     = {1, 8, 32, 128};

        benchmarkMatrixMult(sizes, threads, scheds, chunks, "output/results.csv");
        
        // ─── GENERAZIONE AUTOMATICA GRAFICI ─────────────────────────────────
        std::cout << "\n========================================\n";
        std::cout << "Generazione automatica dei grafici...\n";
        std::cout << "========================================\n";
        
        // Chiamata allo script Python
        int ret = system("python3 plot_results.py");
        
        if (ret == 0) {
            std::cout << "\n✅ Grafici generati con successo nella cartella 'figures/'\n";
            std::cout << "   Includili nel report LaTeX con:\n";
            std::cout << "   \\includegraphics[width=\\textwidth]{figures/fig1_speedup_threads.pdf}\n";
        } else {
            std::cerr << "\n⚠️  Attenzione: generazione grafici fallita\n";
            std::cerr << "   Assicurati di avere installato: pip install pandas matplotlib numpy\n";
        }
    }

    else {
        std::cerr << "Modalità sconosciuta: " << mode << "\n";
        std::cerr << "Uso: ./matmul [test|benchmark]\n";
        return 1;
    }

    return 0;
}