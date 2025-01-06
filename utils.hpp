#pragma once

#include <iostream>

#include <typeinfo>
#include <cxxabi.h>
#include <memory>
#include <chrono>

#include <random>
#include <cmath>

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
                _start = std::chrono::system_clock::now();
            }
            ~Timer(){
                std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end -_start).count();
                std::cout << "Duration spent on " << _task_name << " is " << duration << "\n";
            }

        private:
            std::chrono::system_clock::time_point _start;
            std::string _task_name;


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
            b[i] = 0.0f; 
        }
    }
}
    // Usage:
    //xavier_init(W1_h, INPUT_DIM, HIDDEN_DIM);   

