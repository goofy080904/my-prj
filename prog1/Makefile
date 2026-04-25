CXX      = g++
CXXFLAGS = -O3 -fopenmp -std=c++17 -Wall -Wextra -march=native

SRC_DIR  = src
TEST_DIR = tests
OUT_DIR  = output

SRC      = $(SRC_DIR)/Matrix.cpp $(SRC_DIR)/ParallelMatrixMult.cpp $(SRC_DIR)/main.cpp
TARGET   = matmul
TEST_BIN = test_runner

# ─── Default: compila il binario principale ───────────────────────────────────
all: $(OUT_DIR) $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@

# ─── Crea la cartella output se non esiste ────────────────────────────────────
$(OUT_DIR):
	mkdir -p $(OUT_DIR)

# ─── Test di correttezza (quick, matrice 256x256) ────────────────────────────
test: all
	./$(TARGET) test

# ─── Test unitari (suite completa) ───────────────────────────────────────────
test_unit: $(OUT_DIR) $(TEST_BIN)
	./$(TEST_BIN)

$(TEST_BIN): $(SRC_DIR)/Matrix.cpp $(SRC_DIR)/ParallelMatrixMult.cpp $(TEST_DIR)/test_matrix.cpp
	$(CXX) $(CXXFLAGS) $^ -o $@

# ─── Benchmark completo + GENERAZIONE AUTOMATICA GRAFICI ─────────────────────
benchmark: all
	@echo "🚀 Avvio benchmark..."
	./$(TARGET) benchmark
	@echo ""
	@echo "✅ Benchmark completato! I grafici sono stati generati automaticamente."
	@echo "📁 Risultati CSV: output/results.csv"
	@echo "📊 Grafici: figures/"

# ─── Pulizia ─────────────────────────────────────────────────────────────────
clean:
	rm -f $(TARGET) $(TEST_BIN)
	rm -rf $(OUT_DIR)
	rm -rf figures

.PHONY: all test test_unit benchmark clean check_deps