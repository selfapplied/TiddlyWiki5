/*
 * Zeta Fractal CA WebAssembly Kernel
 * Recursive attractor descent computation for TiddlyWiki
 * 
 * Compile with: emcc -O3 -s WASM=1 -s EXPORTED_FUNCTIONS="['_updateCA', '_insertPrime', '_subdivideCell']" 
 *               -s MODULARIZE=1 -s EXPORT_NAME="ZetaCAModule" zeta_ca_kernel.cpp -o zeta_ca_kernel.js
 */

#include <cmath>
#include <complex>
#include <vector>
#include <algorithm>

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#define EMSCRIPTEN_KEEPALIVE extern "C" __attribute__((used))
#else
#define EMSCRIPTEN_KEEPALIVE
#endif

// Cell structure for CA
struct ZetaCell {
    std::complex<float> value;      // Current zeta approximation
    float moebius_spin;             // Möbius inversion phase
    float error;                    // Spectral approximation error
    int depth;                      // Recursive subdivision depth
    std::complex<float> position;   // Position in complex plane
    float size;                     // Cell size
    bool active;                    // Whether cell is evolving
    int prime_content;              // Number of primes inserted
};

// Global CA state
static std::vector<ZetaCell> ca_cells;
static std::vector<int> primes = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71};
static int current_prime_index = 0;
static int iteration_count = 0;
static float error_threshold = 1e-6f;

// Möbius function
int moebius_mu(int n) {
    if (n == 1) return 1;
    
    int factor_count = 0;
    int temp = n;
    
    for (int p : primes) {
        if (p * p > temp) break;
        
        int count = 0;
        while (temp % p == 0) {
            temp /= p;
            count++;
        }
        
        if (count > 1) return 0;  // Squared prime factor
        if (count == 1) factor_count++;
    }
    
    if (temp > 1) factor_count++;  // Prime factor > sqrt(n)
    
    return (factor_count % 2 == 0) ? 1 : -1;
}

// Euler factor computation: (1 - p^(-s))^(-1)
std::complex<float> euler_factor(int p, std::complex<float> s) {
    float log_p = std::log(static_cast<float>(p));
    std::complex<float> minus_s = -s;
    std::complex<float> p_to_minus_s = std::exp(minus_s * log_p);
    
    std::complex<float> denominator = 1.0f - p_to_minus_s;
    
    // Avoid division by zero
    if (std::abs(denominator) < 1e-15f) {
        return std::complex<float>(1e15f, 0.0f);
    }
    
    return 1.0f / denominator;
}

// Möbius twist transformation
std::complex<float> apply_moebius_twist(const ZetaCell& cell, std::complex<float> s) {
    std::complex<float> a = 1.0f + std::complex<float>(0.0f, 0.1f * cell.moebius_spin);
    std::complex<float> b = std::complex<float>(0.1f * cell.error, 0.0f);
    float c = 0.01f * cell.prime_content;
    float d = 1.0f;
    
    std::complex<float> denominator = c * s + d;
    
    if (std::abs(denominator) < 1e-15f) {
        return s;
    }
    
    return (a * s + b) / denominator;
}

// Initialize CA with single attractor cell
EMSCRIPTEN_KEEPALIVE
void initCA(float real, float imag, int grid_size) {
    ca_cells.clear();
    current_prime_index = 0;
    iteration_count = 0;
    
    ZetaCell initial_cell;
    initial_cell.value = std::complex<float>(1.0f, 0.0f);
    initial_cell.moebius_spin = 0.0f;
    initial_cell.error = 1.0f;
    initial_cell.depth = 0;
    initial_cell.position = std::complex<float>(real, imag);
    initial_cell.size = 2.0f;
    initial_cell.active = true;
    initial_cell.prime_content = 0;
    
    ca_cells.push_back(initial_cell);
}

// Insert prime into all active cells
EMSCRIPTEN_KEEPALIVE
void insertPrime(int prime) {
    for (auto& cell : ca_cells) {
        if (!cell.active) continue;
        
        std::complex<float> s = cell.position;
        std::complex<float> euler = euler_factor(prime, s);
        std::complex<float> old_value = cell.value;
        
        cell.value *= euler;
        
        // Update error estimate
        float error_contribution = std::abs(cell.value - old_value) / std::max(std::abs(cell.value), 1e-15f);
        cell.error = std::max(cell.error * 0.9f, error_contribution);
        
        // Update Möbius spin
        int mu = moebius_mu(prime);
        cell.moebius_spin = 0.7f * cell.moebius_spin + 0.3f * mu;
        
        cell.prime_content++;
    }
}

// Check if cell should subdivide
bool should_subdivide(const ZetaCell& cell) {
    return cell.error > error_threshold && cell.depth < 10 && cell.active;
}

// Subdivide a cell into 4 subcells
EMSCRIPTEN_KEEPALIVE
int subdivideCell(int cell_index) {
    if (cell_index >= ca_cells.size() || !should_subdivide(ca_cells[cell_index])) {
        return 0;  // No subdivision
    }
    
    ZetaCell& parent = ca_cells[cell_index];
    parent.active = false;
    
    float new_size = parent.size * 0.5f;
    std::complex<float> pos = parent.position;
    
    // Create 4 subcells
    std::vector<std::complex<float>> offsets = {
        std::complex<float>(-0.25f, -0.25f) * parent.size,
        std::complex<float>(0.25f, -0.25f) * parent.size,
        std::complex<float>(-0.25f, 0.25f) * parent.size,
        std::complex<float>(0.25f, 0.25f) * parent.size
    };
    
    for (const auto& offset : offsets) {
        std::complex<float> new_pos = pos + offset;
        
        // Apply Möbius twist to new position
        std::complex<float> twisted_pos = apply_moebius_twist(parent, new_pos);
        
        ZetaCell subcell;
        subcell.value = parent.value;
        subcell.moebius_spin = parent.moebius_spin;
        subcell.error = parent.error * 0.8f;  // Assume subdivision improves error
        subcell.depth = parent.depth + 1;
        subcell.position = twisted_pos;
        subcell.size = new_size;
        subcell.active = true;
        subcell.prime_content = parent.prime_content;
        
        ca_cells.push_back(subcell);
    }
    
    return 4;  // Created 4 new cells
}

// Apply Fourier ↔ Möbius coupling between neighboring cells
void apply_coupling() {
    // Simplified coupling for WASM version
    for (size_t i = 0; i < ca_cells.size(); i++) {
        if (!ca_cells[i].active) continue;
        
        // Critical line attraction (Re(s) → 0.5)
        float critical_pull = 0.01f * (0.5f - ca_cells[i].position.real());
        ca_cells[i].position += std::complex<float>(critical_pull, 0.0f);
        
        // Möbius spin evolution
        float evolution_rate = 0.1f * std::sin(iteration_count * 0.1f);
        ca_cells[i].moebius_spin += evolution_rate * moebius_mu(ca_cells[i].prime_content + 1);
    }
}

// Main CA evolution step
EMSCRIPTEN_KEEPALIVE
int updateCA() {
    iteration_count++;
    
    // Insert next prime if available
    if (current_prime_index < primes.size()) {
        insertPrime(primes[current_prime_index]);
        current_prime_index++;
    }
    
    // Check for subdivisions
    std::vector<int> subdivision_candidates;
    for (size_t i = 0; i < ca_cells.size(); i++) {
        if (should_subdivide(ca_cells[i])) {
            subdivision_candidates.push_back(i);
        }
    }
    
    // Perform subdivisions (from back to front to avoid index issues)
    int subdivisions_performed = 0;
    for (auto it = subdivision_candidates.rbegin(); it != subdivision_candidates.rend(); ++it) {
        subdivisions_performed += subdivideCell(*it);
    }
    
    // Apply coupling
    apply_coupling();
    
    return subdivisions_performed;
}

// Get CA state for WebGL texture update
EMSCRIPTEN_KEEPALIVE
void getCellData(float* ca_data, float* prime_data, int width, int height) {
    // Map cells to texture grid
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * 4;
            
            // Find nearest active cell or interpolate
            float u = static_cast<float>(x) / width;
            float v = static_cast<float>(y) / height;
            
            // For simplicity, use first active cell
            // In production, would do proper spatial mapping
            ZetaCell* active_cell = nullptr;
            for (auto& cell : ca_cells) {
                if (cell.active) {
                    active_cell = &cell;
                    break;
                }
            }
            
            if (active_cell) {
                ca_data[idx + 0] = active_cell->value.real();
                ca_data[idx + 1] = active_cell->value.imag();
                ca_data[idx + 2] = active_cell->moebius_spin;
                ca_data[idx + 3] = active_cell->error;
                
                prime_data[idx + 0] = active_cell->prime_content;
                prime_data[idx + 1] = active_cell->depth;
                prime_data[idx + 2] = 0.0f;
                prime_data[idx + 3] = 1.0f;
            } else {
                // Default values
                ca_data[idx + 0] = 1.0f;
                ca_data[idx + 1] = 0.0f;
                ca_data[idx + 2] = 0.0f;
                ca_data[idx + 3] = 1.0f;
                
                prime_data[idx + 0] = 0.0f;
                prime_data[idx + 1] = 0.0f;
                prime_data[idx + 2] = 0.0f;
                prime_data[idx + 3] = 1.0f;
            }
        }
    }
}

// Get statistics
EMSCRIPTEN_KEEPALIVE
int getActiveCellCount() {
    return std::count_if(ca_cells.begin(), ca_cells.end(), 
                        [](const ZetaCell& cell) { return cell.active; });
}

EMSCRIPTEN_KEEPALIVE
float getAverageError() {
    float total_error = 0.0f;
    int active_count = 0;
    
    for (const auto& cell : ca_cells) {
        if (cell.active) {
            total_error += cell.error;
            active_count++;
        }
    }
    
    return active_count > 0 ? total_error / active_count : 0.0f;
}

EMSCRIPTEN_KEEPALIVE
int getCurrentIteration() {
    return iteration_count;
}

EMSCRIPTEN_KEEPALIVE
int getCurrentPrimeIndex() {
    return current_prime_index;
}