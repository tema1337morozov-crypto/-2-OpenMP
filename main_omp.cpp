#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace chrono;

// Функция сохранения матрицы в файл
void saveMatrix(const vector<vector<double>>& matrix, const string& filename) {
    ofstream file(filename);
    if (!file.is_open()) {
        cerr << "ОШИБКА: не могу открыть файл " << filename << " для записи!" << endl;
        return;
    }

    int n = matrix.size();
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            file << matrix[i][j];
            if (j < n - 1) file << " ";
        }
        file << endl;
    }
    file.close();
    cout << "  Файл " << filename << " сохранён (размер " << n << "x" << n << ")" << endl;
}

// Функция чтения матрицы из файла
vector<vector<double>> loadMatrix(const string& filename, int& size) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "ОШИБКА: не могу открыть файл " << filename << " для чтения!" << endl;
        exit(1);
    }

    file >> size;
    vector<vector<double>> matrix(size, vector<double>(size));
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            file >> matrix[i][j];
        }
    }
    file.close();
    return matrix;
}

// ПОСЛЕДОВАТЕЛЬНОЕ умножение
vector<vector<double>> multiplySequential(const vector<vector<double>>& A, const vector<vector<double>>& B) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}

// ПАРАЛЛЕЛЬНОЕ умножение с OpenMP
vector<vector<double>> multiplyParallel(const vector<vector<double>>& A, const vector<vector<double>>& B, int num_threads) {
    int n = A.size();
    vector<vector<double>> C(n, vector<double>(n, 0.0));

    omp_set_num_threads(num_threads);

#pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }

    return C;
}

int main() {
    int n1, n2;

    cout << "========================================" << endl;
    cout << "   Умножение матриц с OpenMP" << endl;
    cout << "========================================" << endl;

    // Читаем матрицы
    cout << "Чтение матриц из файлов..." << endl;
    auto A = loadMatrix("matrixA.txt", n1);
    auto B = loadMatrix("matrixB.txt", n2);

    if (n1 != n2) {
        cerr << "Ошибка: размеры матриц не совпадают (" << n1 << " != " << n2 << ")" << endl;
        return 1;
    }
    int n = n1;

    cout << "Размер матрицы: " << n << " x " << n << endl;
    cout << "Доступно ядер: " << omp_get_num_procs() << endl;
    cout << "----------------------------------------" << endl;

    // Сначала проверим последовательное умножение и сохраним результат
    cout << "\n[1] Последовательное умножение..." << endl;
    auto start_seq = high_resolution_clock::now();
    vector<vector<double>> C_seq = multiplySequential(A, B);
    auto end_seq = high_resolution_clock::now();
    auto time_seq = duration_cast<milliseconds>(end_seq - start_seq).count();
    double sec_seq = time_seq / 1000.0;

    // СОХРАНЯЕМ РЕЗУЛЬТАТ ПОСЛЕДОВАТЕЛЬНОГО УМНОЖЕНИЯ
    saveMatrix(C_seq, "result_seq.txt");

    long long ops = (long long)n * n * n;
    cout << "  Время: " << sec_seq << " сек" << endl;
    cout << "  MFLOPS: " << (ops / 1e6) / sec_seq << endl;

    // Теперь тесты с разным количеством потоков
    cout << "\n[2] Параллельное умножение (OpenMP)..." << endl;

    vector<int> threads_list = { 1, 2, 4, 8 };
    vector<double> times;

    for (int t : threads_list) {
        cout << "\n  Тест с " << t << " поток(ами)..." << endl;

        auto start = high_resolution_clock::now();
        vector<vector<double>> C_par = multiplyParallel(A, B, t);
        auto end = high_resolution_clock::now();

        auto duration = duration_cast<milliseconds>(end - start);
        double seconds = duration.count() / 1000.0;
        times.push_back(seconds);

        // СОХРАНЯЕМ РЕЗУЛЬТАТ ДЛЯ ПЕРВОГО ПОТОКА (как эталон)
        if (t == 1) {
            saveMatrix(C_par, "result_omp.txt");
        }

        double mflops = (ops / 1000000.0) / seconds;
        cout << "    Время: " << seconds << " сек" << endl;
        cout << "    MFLOPS: " << mflops << endl;
    }

    // Вывод сводной таблицы
    cout << "\n========================================" << endl;
    cout << "СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ" << endl;
    cout << "========================================" << endl;
    cout << "Потоки\tВремя (сек)\tУскорение" << endl;

    double sequential_time = times[0];
    for (size_t i = 0; i < threads_list.size(); i++) {
        double speedup = sequential_time / times[i];
        cout << threads_list[i] << "\t" << times[i] << "\t\t" << speedup << endl;
    }

    cout << "\nРезультаты сохранены в файлы:" << endl;
    cout << "  - result_seq.txt (последовательное умножение)" << endl;
    cout << "  - result_omp.txt (параллельное умножение, 1 поток)" << endl;

    return 0;
}