#include<vector>
#include<string>
#include<iostream>
#include<algorithm>
#include <climits>
#include<math.h>

using namespace std;

#define DEBUG_OPTION 1
#define ALIGNDOWN(a, b) ((a / b) * b)
#define CAL_OUT(a, b, c) ((a - b) / c)

namespace cobra {
/*----------- mdspan struct ---------*/
/* 
 * A simple multidimensional span-like structure to hold a pointer to data
 * and its shape (dimensions).
 */
template<typename T>
struct mdspan {
    T *data;
    std::vector<int> shape;
    mdspan(T* data, const std::vector<int>& shape) : data(data), shape(shape) {}
};

/*------- To initialize data ------*/
/* Initialize the vector with incrementing values starting from 1 */
template<typename T>
void initialize(std::vector<T>& data) {
    int size = data.size();
    for (int i = 0; i < size; i++) {
        data[i] = i + 1;
    }
} // initialize

/*------- To Calc Stride -------*/
/* 
 * Calculate strides for each dimension of the tensor.
 * Stride indicates the number of elements to skip to reach the next element
 * along a specific dimension.
 */
template<typename T>
auto cal_stride(const std::vector<T>& shape) {
    vector<T> stride(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; --i) {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
    return stride;
} // cal_stride

/*-------- Debug print functions ---------*/
/* Utility function to print 1D vectors */
template<typename T>
void print(const std::vector<T>& arr, bool flag = 0) {
    if (DEBUG_OPTION || flag) {
        std::cout << std::endl;
        for (auto i : arr) {
            std::cout << i << ", ";
        }
        std::cout << std::endl;
    }
} // print

/* Overload for printing single values */
template<typename T>
void print(const T& s) {
    if (DEBUG_OPTION) {
        std::cout << std::endl << s << std::endl;
    }
} // print

// Special case for printing raw arrays with a size limit
namespace sp {
template<typename T>
void print(const T *arr) {
    if (DEBUG_OPTION) {
        int size = 2;  // Limiting size for demo purposes
        std::cout << std::endl;
        for (auto i = 0; i <size; i++) {
            std::cout << arr[i] << ", ";
        }
        std::cout << std::endl;
    }
} // print
}

/*----- To Print Multi-Dimensional array ------*/
/* Recursively display multi-dimensional data using strides for indexing */
template<typename T>
void dis(int dim, int idx, T *data, const std::vector<int>& st,
         const std::vector<int>& sh, int rank) {
    if (dim == rank) {
        std::cout << data[idx] << ", ";
        return;
    }

    std::cout << "[ ";
    for (int i = 0; i < sh[dim]; i++) {
        dis(dim + 1, idx + st[dim] * i, data, st, sh, rank);
    }
    std::cout << "]\n";
} // dis

/* Print the shape and data of an mdspan object */
template<typename T>
void print(mdspan<T> *ob, bool flag = 0, string s = "") {
    if (DEBUG_OPTION || flag) {
        std::cout << s;
        std::cout << "\nShape: \n";
        for (auto i : ob->shape) {
            std::cout << i << ", ";
        }
        std::cout << "\nDATA: \n";
        dis(0, 0, ob->data, cal_stride(ob->shape), ob->shape, ob->shape.size());
        std::cout << std::endl;
    }
} // print

/*--------- Copy function for slicing ------------*/
/*
 * Copy subregions of one multi-dimensional array to another
 * using stride-based indexing. This allows for slicing the tensor.
 */
template<typename T1, typename T2>
void cpy(int dim, int dst, int src, const vector<T1>& dst_st, const vector<T1>& src_st,
         const vector<T1>& dst_sh, const vector<T1>& src_sh, int rank, T2 *out, T2 *in,
         const vector<int>& src_offset, const vector<int>& dst_offset) {
    if (dim == rank) {
        out[dst] = in[src];
        return;
    }
    for (int i = dst_offset[dim], j = src_offset[dim];
         i < dst_sh[dim] && j < src_sh[dim]; i++, j++) {
        cpy(dim + 1, dst + dst_st[dim] * i, src + src_st[dim] * j,
            dst_st, src_st, dst_sh, src_sh, rank, out, in, src_offset, dst_offset);
    }
} // cpy

/*--------- Slice function ----------*/
/*
 * Perform slicing on multi-dimensional arrays using offsets and strides.
 * This will copy a subregion from the source array to the destination.
 */
template<typename T>
void slice(mdspan<T> *des, mdspan<T> *source,
           vector<int> dst_offset = {}, vector<int> src_offset = {}) {
    auto des_st = cal_stride(des->shape);
    auto src_st = cal_stride(source->shape);
    int rank = des_st.size();

    if (dst_offset.empty()) {
        dst_offset.resize(rank, 0);
    }
    if (src_offset.empty()) {
        src_offset.resize(rank, 0);
    }

    cpy(0, 0, 0, des_st, src_st, des->shape, source->shape, rank,
        des->data, source->data, src_offset, dst_offset);
} // slice

} // namespace cobra

struct op_para {
    bool align_corner;
    float scale_factor_row;
    float scale_factor_col;
};

template<typename T>
void cfunc(T *out, T *in, int n, int c, int w,
           int out_idy, int in_idy, int out_idx,
           int in_idx, int c_in, int w_in, op_para para) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < c; j++) {
            float idx_y = (j + out_idy) * para.scale_factor_row;
            int low_y = floor(idx_y);
            float weight_y = idx_y - low_y;
            low_y -= in_idy;
            int high_y = min(ceil(idx_y) - in_idy, c_in * 1.0f);
            for (int k = 0; k < w; k++) {
                float idx_x = (k + out_idx) * para.scale_factor_col;
                int low_x = floor(idx_x);
                float weight_x = idx_x - low_x;
                low_x -= in_idx;
                int high_x = min(ceil(idx_x) - in_idx, w_in * 1.0f);

                T r_up = in[i * (c_in * w_in) + low_y * w_in + low_x];
                T l_up = in[i * (c_in * w_in) + low_y * w_in + high_x];
                T r_dn = in[i * (c_in * w_in) + high_y * w_in + low_x];
                T l_dn = in[i * (c_in * w_in) + high_y * w_in + high_x];

                float val_1 = (r_up * (1.0f - weight_x) +  l_up * (weight_x));
                float val_2 = (r_dn * (1.0f - weight_x) + l_dn * (weight_x));
                out[i * (c * w) + j * (w) + k] = (val_1 * (1.0f - weight_y) + val_2 * weight_y);
            }
        }
    }
}

template<typename T>
void cfunc_align_false(T *out, T *in, int n, int c, int w,
           int out_idy, int in_idy, int out_idx,
           int in_idx, int c_in, int w_in, op_para para) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < c; j++) {
            float idx_y = (j + out_idy + 0.5f) * para.scale_factor_row - 0.5f;
            int low_y = max(floor(idx_y), 0.0f);
            // cout<<"\n-->"<<idx_y<<" "<<low_y<<endl;
            float weight_y = idx_y - low_y;
            // cout<<weight_y;
            low_y -= in_idy;
            int high_y = min(ceil(idx_y) - in_idy, (c_in - 1) * 1.0f);
            for (int k = 0; k < w; k++) {
                float idx_x = (k + out_idx + 0.5f) * para.scale_factor_col - 0.5f;
                int low_x = max(floor(idx_x), in_idx * 1.0f);
                float weight_x = idx_x - low_x;
                low_x -= in_idx;
                int high_x = min(ceil(idx_x) - in_idx, (w_in - 1) * 1.0f);

                // cout<<"\nweight "<<low_y<<" "<<high_y<<" "<<low_x<<" "<<high_x<<endl;

                T r_up = in[i * (c_in * w_in) + low_y * w_in + low_x];
                T l_up = in[i * (c_in * w_in) + low_y * w_in + high_x];
                T r_dn = in[i * (c_in * w_in) + high_y * w_in + low_x];
                T l_dn = in[i * (c_in * w_in) + high_y * w_in + high_x];
                // cout<<"\n"<<r_up<<" "<<l_up<<" "<<r_dn<<" "<<l_dn<<endl;

                float val_1 = (r_up * (1.0f - weight_x) +  l_up * (weight_x));
                float val_2 = (r_dn * (1.0f - weight_x) + l_dn * (weight_x));
                out[i * (c * w) + j * (w) + k] = (val_1 * (1.0f - weight_y) + val_2 * weight_y);
            }
        }
    }
}

int main() {
    using namespace cobra;
    // Initializing input and output buffers
    vector<float> in_l1_mem(100000, 0);
    vector<float> src(1250000);
    vector<float> out_l1_mem(1000000, INT_MIN); // INT_MIN is used to track uninitialized areas
    vector<float> dst(1250000);

    initialize(src);  // Initialize input with sequential values

    op_para para;

    para.align_corner = 0;
    /*------------------- SHAPES -------------------------*/
    vector<int> input_g_shape = {1, 2, 4};
    vector<int> input_l_shape = {1, 2, 5};
    vector<int> output_g_shape = {1, 2, 10};
    vector<int> output_l_shape = {1, 2, 10};

    /*-------------------- MDSPAN -----------------------*/
    mdspan<float> src_g{src.data(), input_g_shape};
    mdspan<float> dst_g{dst.data(), output_g_shape};

    
    int *in_ptr = input_g_shape.data();
    int *ptr = output_g_shape.data();
    int *out = output_l_shape.data();

    if (para.align_corner) {
        para.scale_factor_row = abs((in_ptr[1] * 1.0f - 1) / (max(output_g_shape[1] - 1, 1))); // ek gap ko 2 gap pe krna hai
        para.scale_factor_col = abs((in_ptr[2] * 1.0f - 1) / (max(output_g_shape[2] - 1, 1))); // ek gap ko 2 gap pe krna hai
        cout<<para.scale_factor_row<<" "<<para.scale_factor_col<<endl;
        for (int i = 0; i < ptr[0]; i+= out[0]) {
            int rem_n = ptr[0] - i >= out[0] ? out[0]: ptr[0] - i;
            for (int j = 0; j < ptr[1]; j += out[1]) {
                int rem_c = ptr[1] - j >= out[1] ? out[1]: ptr[1] - j;
                float low_idx_y = j * para.scale_factor_row;
                float high_idx_y = (j + rem_c - 1) * para.scale_factor_row;
                int low_y = max(0.0f, floor(low_idx_y));
                int high_y = ceil(high_idx_y);
                // cout<<low_y<<high_y<<endl;

                high_y = min(high_y, in_ptr[1] - 1);
                for (int k = 0; k < ptr[2]; k += out[2]) {
                    int rem_w = ptr[2] - k >= out[2] ? out[2]: ptr[2] - k;
                    float low_idx_x = k * para.scale_factor_col;
                    float high_idx_x = (k + rem_w - 1) * para.scale_factor_col;
                    int low_x = max(0.0f, floor(low_idx_x));
                    int high_x = ceil(high_idx_x);
                    
                    // cout<<low_y<<high_y;
                    
                    high_x = min(high_x, in_ptr[2] - 1);
                    cobra::mdspan<float>in_l(in_l1_mem.data(), {rem_n, (high_y - low_y + 1), (high_x - low_x + 1)});
                    cobra::slice(&in_l, &src_g, {0, 0, 0}, {i, low_y, low_x});
                    // print(&in_l);

                    cfunc(out_l1_mem.data(), in_l.data, rem_n, rem_c, rem_w, j, low_y, k, low_x,
                          (high_y - low_y + 1), (high_x - low_x + 1), para);
                    
                    cobra::mdspan<float>out_l(out_l1_mem.data(), {rem_n, rem_c, rem_w});
                    cobra::slice(&dst_g, &out_l, {i, j, k}, {0, 0, 0});
                }
            }
        }
    } else {
        para.scale_factor_row = (in_ptr[1] * 1.0f) / (output_g_shape[1]); // ek gap ko 2 gap pe krna hai
        para.scale_factor_col = (in_ptr[2] * 1.0f) / (output_g_shape[2]); // ek gap ko 2 gap pe krna hai
        // cout<<para.scale_factor_row<<" "<<para.scale_factor_col<<endl;
        for (int i = 0; i < ptr[0]; i+= out[0]) {
            int rem_n = ptr[0] - i >= out[0] ? out[0]: ptr[0] - i;
            for (int j = 0; j < ptr[1]; j += out[1]) {
                int rem_c = ptr[1] - j >= out[1] ? out[1]: ptr[1] - j;
                float low_idx_y = (j + 0.5f) * para.scale_factor_row - 0.5f;
                float high_idx_y = (j + rem_c - 1 + 0.5f) * para.scale_factor_row - 0.5f;
                int low_y = max(0.0f, floor(low_idx_y));
                int high_y = ceil(high_idx_y);
                // cout<<low_y<<high_y<<endl;

                high_y = min(high_y, in_ptr[1] - 1);
                for (int k = 0; k < ptr[2]; k += out[2]) {
                    int rem_w = ptr[2] - k >= out[2] ? out[2]: ptr[2] - k;
                    float low_idx_x = (k + 0.5f) * para.scale_factor_col - 0.5f;
                    float high_idx_x = (k + rem_w - 1 + 0.5f) * para.scale_factor_col - 0.5f;
                    int low_x = max(0.0f, floor(low_idx_x));
                    int high_x = ceil(high_idx_x);
                    
                    // cout<<low_y<<high_y;
                    
                    high_x = min(high_x, in_ptr[2] - 1);
                    cobra::mdspan<float>in_l(in_l1_mem.data(), {rem_n, (high_y - low_y + 1), (high_x - low_x + 1)});
                    cobra::slice(&in_l, &src_g, {0, 0, 0}, {i, low_y, low_x});
                    // print(&in_l);

                    cfunc_align_false(out_l1_mem.data(), in_l.data, rem_n, rem_c, rem_w, j, low_y, k, low_x,
                          (high_y - low_y + 1), (high_x - low_x + 1), para);
                    
                    cobra::mdspan<float>out_l(out_l1_mem.data(), {rem_n, rem_c, rem_w});
                    cobra::slice(&dst_g, &out_l, {i, j, k}, {0, 0, 0});
                }
            }
        }
    }

    // print(&src_g);
    print(&dst_g);
    return 0;
}
