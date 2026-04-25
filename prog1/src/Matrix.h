#ifndef MATRIX_H
#define MATRIX_H

#include <cstddef>
#include <string>

class Matrix {
private:
    float*  data;
    size_t  rows;
    size_t  cols;

public:
    Matrix(size_t rows, size_t cols);
    ~Matrix();

    // Nessuna copia: move-only per evitare copie accidentali di grandi buffer
    Matrix(const Matrix&)            = delete;
    Matrix& operator=(const Matrix&) = delete;
    Matrix(Matrix&& other) noexcept;
    Matrix& operator=(Matrix&& other) noexcept;

    // Accesso inline cache-friendly (row-major)
    inline float&       at(size_t i, size_t j)       { return data[i * cols + j]; }
    inline const float& at(size_t i, size_t j) const { return data[i * cols + j]; }

    // Accesso raw per ottimizzazioni avanzate (pointer-arithmetics)
    inline float*       raw()       { return data; }
    inline const float* raw() const { return data; }

    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }

    void randomInit(unsigned int seed = 42);
    void zeros();
    bool isEqual(const Matrix& other, float epsilon = 1e-2f) const;

    void saveToFile(const std::string& filename) const;
    void loadFromFile(const std::string& filename);
};

#endif // MATRIX_H
