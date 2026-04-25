#include "Matrix.h"
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <random>
#include <cmath>

// ─── Costruttore: alloca e azzera il buffer ──────────────────────────────────
Matrix::Matrix(size_t rows, size_t cols)
    : data(new float[rows * cols]()), rows(rows), cols(cols) {}

// ─── Distruttore ─────────────────────────────────────────────────────────────
Matrix::~Matrix() {
    delete[] data;
}

// ─── Move constructor ────────────────────────────────────────────────────────
Matrix::Matrix(Matrix&& other) noexcept
    : data(other.data), rows(other.rows), cols(other.cols) {
    other.data = nullptr;
    other.rows = 0;
    other.cols = 0;
}

// ─── Move assignment ─────────────────────────────────────────────────────────
Matrix& Matrix::operator=(Matrix&& other) noexcept {
    if (this != &other) {
        delete[] data;
        data       = other.data;
        rows       = other.rows;
        cols       = other.cols;
        other.data = nullptr;
        other.rows = 0;
        other.cols = 0;
    }
    return *this;
}

// ─── Inizializzazione casuale con seed fisso per riproducibilità ──────────────
void Matrix::randomInit(unsigned int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < rows * cols; ++i)
        data[i] = dist(rng);
}

// ─── Azzeramento del buffer ───────────────────────────────────────────────────
void Matrix::zeros() {
    std::memset(data, 0, rows * cols * sizeof(float));
}

// ─── Confronto con tolleranza floating-point ──────────────────────────────────
bool Matrix::isEqual(const Matrix& other, float epsilon) const {
    if (rows != other.rows || cols != other.cols) return false;
    for (size_t i = 0; i < rows * cols; ++i)
        if (std::abs(data[i] - other.data[i]) > epsilon) return false;
    return true;
}

// ─── Salvataggio binario (escluso dalla misurazione delle performance) ────────
void Matrix::saveToFile(const std::string& filename) const {
    std::ofstream f(filename, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot write: " + filename);
    f.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    f.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    f.write(reinterpret_cast<const char*>(data), rows * cols * sizeof(float));
}

// ─── Lettura binaria ──────────────────────────────────────────────────────────
void Matrix::loadFromFile(const std::string& filename) {
    std::ifstream f(filename, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot read: " + filename);
    f.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    f.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    delete[] data;
    data = new float[rows * cols];
    f.read(reinterpret_cast<char*>(data), rows * cols * sizeof(float));
}
