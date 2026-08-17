#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <queue>
#include <cmath>
#include <map>
#include <string>

namespace py = pybind11;

// Helper function to verify grid bounds
inline bool is_valid(int r, int c, int rows, int cols) {
    return r >= 0 && r < rows && c >= 0 && c < cols;
}

std::vector<std::map<std::string, double>> find_forged_components(
    py::array_t<double> z_score_grid, double threshold) {
    
    // 1. Request direct memory access to the NumPy buffer
    py::buffer_info buf_info = z_score_grid.request();
    
    if (buf_info.ndim != 2) {
        throw std::runtime_error("Input grid must be exactly 2-dimensional.");
    }
    
    int rows = buf_info.shape[0];
    int cols = buf_info.shape[1];
    
    // 2. Extract raw pointer to the data array for zero-copy access
    double* ptr = static_cast<double*>(buf_info.ptr);
    
    // 3. Flat 1D vector for visited tracking (faster than vector<vector>)
    std::vector<bool> visited(rows * cols, false);
    std::vector<std::map<std::string, double>> components;
    
    // 4-way directional shifts: Up, Down, Left, Right
    int dr[] = {-1, 1, 0, 0};
    int dc[] = {0, 0, -1, 1};
    
    // 4. Receptive Field Scanning
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int idx = i * cols + j;
            
            if (!visited[idx] && std::abs(ptr[idx]) > threshold) {
                // Found an unvisited anomaly. Launch BFS.
                std::queue<std::pair<int, int>> q;
                q.push({i, j});
                visited[idx] = true;
                
                int area = 0;
                int perimeter = 0;
                
                while (!q.empty()) {
                    auto [r, c] = q.front();
                    q.pop();
                    
                    area++;
                    
                    // Check neighbors to build perimeter and expand graph
                    for (int d = 0; d < 4; ++d) {
                        int nr = r + dr[d];
                        int nc = c + dc[d];
                        
                        if (is_valid(nr, nc, rows, cols)) {
                            int n_idx = nr * cols + nc;
                            bool is_anomalous = std::abs(ptr[n_idx]) > threshold;
                            
                            if (is_anomalous) {
                                if (!visited[n_idx]) {
                                    visited[n_idx] = true;
                                    q.push({nr, nc});
                                }
                            } else {
                                // Neighbor is natural noise -> this is an edge
                                perimeter++;
                            }
                        } else {
                            // Image boundary -> this is an edge
                             perimeter++;
                        }
                    }
                }
                
                // 5. Compute compactness ratio
                double compactness = (area > 0) ? (double)perimeter / area : 0.0;
                
                std::map<std::string, double> metrics;
                metrics["area"] = area;
                metrics["perimeter"] = perimeter;
                metrics["compactness"] = compactness;
                
                components.push_back(metrics);
            }
        }
    }
    
    return components;
}

// 6. Pybind11 Python Binding
PYBIND11_MODULE(graph_segmenter_cpp, m) {
    m.doc() = "High-performance C++ Graph Anomaly Segmenter using BFS";
    m.def("find_forged_components", &find_forged_components, 
          "Executes O(V+E) BFS traversal on Z-score grids to find contiguous forged regions.",
          py::arg("z_score_grid"), py::arg("threshold") = 2.5);
}