#pragma once

#include <iostream>

#include <typeinfo>
#include <cxxabi.h>
#include <memory>
#include <chrono>

#include <random>
#include <cmath>

#include <iomanip>
#include <sstream>

//using namespace std::chrono

namespace utils {
    using namespace std::chrono;

    template <typename T>
    void print_it(const T &t, const std::string& sep = " "){
        for (const auto& x:t) std::cout << x << sep;
    }

    std::string type_name(const std::type_info &type){
        int status;
        std::unique_ptr<char, void(*)(void*)> result(
            abi::__cxa_demangle(type.name(), nullptr, nullptr, &status),
            std::free
        );
        return (status == 0) ? result.get() : type.name();
    }

    class Timer {
        public:
            Timer(std::string name): _task_name(name) {
                _start = std::chrono::high_resolution_clock::now();
            }

            //std::string report() {
            //    auto end = std::chrono::high_resolution_clock::now();
            //    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - _start).count();

            //    std::ostringstream ss;
            //    ss << std::fixed << std::setprecision(3);
            //    ss << std::left << std::setw(30) << _task_name + ": ";

            //    if (duration >= 1000000) {
            //        ss << duration/1000000.0 << " s";
            //    } 
            //    else if (duration >= 1000) {
            //        ss << duration/1000.0 << " ms";
            //    }
            //    else {
            //        ss << duration << " μs";
            //    }

            //    return ss.str();
            //}

            std::string report() {
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - _start).count();

                std::string formatted;

                std::ostringstream formattedF;
                formattedF << std::fixed << std::setprecision(3) ;
                formattedF << std::left << std::setw(30) << _task_name;

                auto dur_str = std::to_string(duration);
                int len = dur_str.length();
                
                for (int i = 0; i < len; i++) {
                    formatted += dur_str[i];
                    if (i < len - 1 && (len - i - 1) % 3 == 0) {
                        formatted += '_';
                    }
                }
                formattedF << formatted << " μs";

                return formattedF.str();
            }

            void reset(std::string newTaskName) {
                _start = std::chrono::high_resolution_clock::now();
                _task_name = newTaskName;
            }

        private:
            std::string _task_name;
            std::chrono::high_resolution_clock::time_point _start;
        };

    std::string formatTime(float milliseconds) {
        std::ostringstream formatted;
        formatted << std::fixed << std::setprecision(3) ;
        
        std::string value;
        if (milliseconds >= 1000.0f) {
            value = std::to_string(static_cast<long long>(milliseconds/1000.0f));
            // Add underscores
            std::string result;
            int len = value.length();
            for (int i = 0; i < len; i++) {
                result += value[i];
                if (i < len - 1 && (len - i - 1) % 3 == 0) {
                    result += '_';
                }
            }
            formatted << result << " s";
        }
        else if (milliseconds >= 1.0f) {
            value = std::to_string(static_cast<long long>(milliseconds));
            std::string result;
            int len = value.length();
            for (int i = 0; i < len; i++) {
                result += value[i];
                if (i < len - 1 && (len - i - 1) % 3 == 0) {
                    result += '_';
                }
            }
            formatted << result << " ms";
        }
        else {
            value = std::to_string(static_cast<long long>(milliseconds * 1000.0f));
            std::string result;
            int len = value.length();
            for (int i = 0; i < len; i++) {
                result += value[i];
                if (i < len - 1 && (len - i - 1) % 3 == 0) {
                    result += '_';
                }
            }
            formatted << result << " μs";
        }
        
        return formatted.str();
    }

    void xavier_init(float* W, float *b, int input_dim, int output_dim) {

        float scale = sqrt(6.0f / (input_dim + output_dim));
        
        //std::random_device rd;
        std::mt19937 gen(17);
        std::uniform_real_distribution<float> dis(-scale, scale);
        
        for (int i = 0; i < input_dim * output_dim; i++) {
            W[i] = dis(gen);
        }

        for (int i = 0; i < output_dim; i++) {
            b[i] = 0.0f; 
        }
    }
    
    void cpuMult(const float* A, const float* B, float* C, int n)
    {
        for (int r = 0; r < n; ++r) {
            for (int c = 0; c < n; ++c) {
                float sum = 0.0f;
                for (int k = 0; k < n; ++k) {
                    sum += A[r * n + k] * B[k * n + c];
                }
                C[r * n + c] = sum;
            }
        }
    }

    bool loadMatrix(const std::string &filename,
                    std::vector<float> &weights,
                    int rows,
                    int cols)
    {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return false;
        }

        // (Optional) Ensure the vector has the right size
        if (static_cast<int>(weights.size()) != rows * cols) {
            weights.resize(rows * cols);
        }

        // Read the file line-by-line
        for (int r = 0; r < rows; ++r) {
            for (int c = 0; c < cols; ++c) {
                if (!(file >> weights[r * cols + c])) {
                    std::cerr << "Error reading data at row " << r
                            << ", col " << c << std::endl;
                    return false;
                }
            }
        }

        return true;
    }

}
    // Usage:
    //xavier_init(W1_h, INPUT_DIM, HIDDEN_DIM);   

