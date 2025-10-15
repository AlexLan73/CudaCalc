/**
 * @file fft16_cpu_reference.cpp  
 * @brief CPU reference FFT16 for debugging
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <complex>
#include <cmath>

using Complex = std::complex<float>;

// Bit reverse for 4 bits
int bitReverse4(int x) {
    int result = 0;
    for (int i = 0; i < 4; ++i) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

// Simple CPU FFT16
void fft16_cpu(const std::vector<Complex>& input, std::vector<Complex>& output) {
    const int N = 16;
    output.resize(N);
    
    // Bit-reversal permutation
    std::cout << "Step 1: Bit-reversal permutation\n";
    for (int i = 0; i < N; ++i) {
        int rev = bitReverse4(i);
        output[rev] = input[i];
        if (i < 4) {
            std::cout << "  input[" << i << "] -> output[" << rev << "]\n";
        }
    }
    std::cout << "\n";
    
    // Butterfly stages
    for (int stage = 0; stage < 4; ++stage) {  // log2(16) = 4
        int m = 1 << (stage + 1);  // size of sub-FFT
        int m2 = m / 2;            // half size
        
        Complex W_m = std::exp(Complex(0, -2.0f * M_PI / m));
        
        std::cout << "Stage " << stage << " (m=" << m << ", m/2=" << m2 << "):\n";
        
        for (int k = 0; k < N; k += m) {
            Complex W = 1.0f;
            for (int j = 0; j < m2; ++j) {
                int idx1 = k + j;
                int idx2 = k + j + m2;
                
                Complex t = W * output[idx2];
                Complex u = output[idx1];
                
                output[idx1] = u + t;
                output[idx2] = u - t;
                
                if (k == 0 && j < 2) {
                    std::cout << "  [" << idx1 << "] and [" << idx2 << "]: twiddle angle = " 
                              << std::arg(W) * 180.0f / M_PI << " deg\n";
                }
                
                W *= W_m;
            }
        }
        std::cout << "\n";
    }
}

int main() {
    std::cout << "\n=== CPU FFT16 REFERENCE ===\n\n";
    
    // exp(2πi*n/16)
    std::vector<Complex> input(16);
    for (int i = 0; i < 16; ++i) {
        float angle = 2.0f * M_PI * i / 16.0f;
        input[i] = Complex(std::cos(angle), std::sin(angle));
    }
    
    std::vector<Complex> output;
    fft16_cpu(input, output);
    
    std::cout << "Final output (non-zero bins):\n";
    for (int i = 0; i < 16; ++i) {
        float mag = std::abs(output[i]);
        if (mag > 0.1f) {
            std::cout << "  [" << i << "] = " << output[i].real() << " + " 
                      << output[i].imag() << "i  (mag=" << mag << ")\n";
        }
    }
    
    std::cout << "\nExpected: bin[1] = ~16\n\n";
    
    return 0;
}

