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

            void report() {
                auto end = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - _start).count();

                std::string formatted;
                auto dur_str = std::to_string(duration);
                int len = dur_str.length();
                
                for (int i = 0; i < len; i++) {
                    formatted += dur_str[i];
                    if (i < len - 1 && (len - i - 1) % 3 == 0) {
                        formatted += '_';
                    }
                }

                std::cout << _task_name << ": " << formatted << " μs" << std::endl;
            }

        private:
            std::string _task_name;
            std::chrono::high_resolution_clock::time_point _start;
        };

    void xavier_init(float* W, float *b, int input_dim, int output_dim) {

        float scale = sqrt(6.0f / (input_dim + output_dim));
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-scale, scale);
        
        for (int i = 0; i < input_dim * output_dim; i++) {
            W[i] = dis(gen);
        }

        for (int i = 0; i < output_dim; i++) {
            b[i] = i; 
            //b[i] = 0.0f; 
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





}
    // Usage:
    //xavier_init(W1_h, INPUT_DIM, HIDDEN_DIM);   

