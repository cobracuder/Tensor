#include <bits/stdc++.h>
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
           vector<int> dst_offset = {}, vector<int> src_offset = {},
           vector<int> des_st = {}, vector<int> src_st = {}) {
    des_st = cal_stride(des->shape);
    src_st = cal_stride(source->shape);
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

template<typename T1, typename T2>
void cpy2(int dim, int dim2, int dst, int src, const vector<T1>& dst_st, const vector<T1>& src_st,
         const vector<T1>& dst_sh, const vector<T1>& src_sh, int rank, T2 *out, T2 *in,
         const vector<int>& src_offset, const vector<int>& dst_offset) {

    if (dst_st[dim])
    if (dim == rank) {
        out[dst] = in[src];
        return;
    }


    for (int i = dst_offset[dim], j = src_offset[dim2];
         i < dst_sh[dim] && j < src_sh[dim2]; i++, j++) {
        cpy2(dim + 1, dst + dst_st[dim] * i, src + src_st[dim] * j,
            dst_st, src_st, dst_sh, src_sh, rank, out, in, src_offset, dst_offset);
    }
} // cpy

template<typename T>
void strided(int dim, int64_t index, vector<int64_t>strides, vector<int64_t>shape, vector<T>&dst, T *src) {
    if(dim == shape.size()){
        dst.push_back(src[index]);
        return;
    }

    for (int i = 0; i < shape[dim]; i++){
        strided(dim + 1, index + strides[dim] * i , strides, shape, dst, src);
    }
} // slice
} // namespace cobra

/*--------- Max Pool function ----------*/
/* Define pooling parameters in a structure for clarity */
struct op_para {
    int kernel[3];         // Kernel size
    bool ceil_mode;        // Ceil mode flag for pooling
    int stride[2];         // Stride for pooling
    int dilation[2];       // Dilation factor for pooling
    int dilated_kernel[2]; // Dilated kernel size
    int padding[4];        // Padding values
};

/* 
 * Max Pooling function: Perform max pooling operation
 * on input tensor with stride, dilation, and padding.
 */
template<typename T>
void maxpool_cfunc(T *out, T *in, int w, int h, int c, int dilated_kernel,
                   int stride, int dilation, int kernel, op_para para) {
    int sliding = ceil((h - dilated_kernel + 1.0f) / stride);
    for (int j = 0; j < w; j++) {
        for (int k = 0, out_idx = 0, in_idx = 0; k < sliding; k += 1, out_idx += stride, in_idx += kernel) {
            for (int i = 0; i < c; i++) {
                // Apply the kernel with dilation over the input
                for (int d_l = 0, in_l = 0; d_l < dilated_kernel; d_l += dilation, in_l++) {
                    out[(out_idx + d_l) * c + i] += in[(in_idx + in_l) * c + i];
                }
            }
        }
    }
} // maxpool_cfunc

/******************************************************************************************/
void set_op_para(op_para &para) {
    para.kernel[0] = 3;
    para.kernel[1] = 3;
    para.ceil_mode = 0;    // Enable ceil mode for pooling
    para.stride[0] = 2;
    para.stride[1] = 2;
    para.dilation[0] = 1;
    para.dilation[1] = 1;
    para.dilated_kernel[0] = para.dilation[0] * (para.kernel[0] - 1) + 1;
    para.dilated_kernel[1] = para.dilation[1] * (para.kernel[1] - 1) + 1;
    para.padding[0] = 0;
    para.padding[1] = 0;
}

void get_in_shape(vector<int>&in_shape, int W, int H, int C, op_para para) {
    in_shape[0] = W;
    in_shape[1] = ceil((H - para.dilated_kernel[0] + 1.0f) / para.stride[0]);
    in_shape[2] = ceil((in_shape[1] - para.dilated_kernel[0] + 1.0f) / para.stride[0]);
}

void set_in_offset_after_padding(int &l1_offset, int &global_offset,
                                 int pad_size, int cur_in_idx) {
    l1_offset = cur_in_idx < pad_size? pad_size - cur_in_idx: 0;
    global_offset = std::max(cur_in_idx - pad_size, 0);
}

int main() {
    using namespace cobra;
    // Initializing input and output buffers
    vector<int> in_l1_mem(100000, 0);
    vector<int> src(1250000);
    vector<int> out_l1_mem(1000000, 0); // INT_MIN is used to track uninitialized areas
    vector<int> dst(1250000);

    initialize(src);  // Initialize input with sequential values

    /*------------------ INITIALIZE PARAMETERS ----------------*/
    op_para para;
    set_op_para(para);

    /*------------------- SHAPES -------------------------*/
    vector<int> input_l_shape = {1, 2, 6};
    vector<int> output_l_shape = {1, 4, 6};
    vector<int> output_g_shape = {1, 9, 6};

    int sliding_window = ceil((output_g_shape[1] - para.dilated_kernel[0] + 1.0f) / para.stride[0]);
    vector<int> input_g_shape = {1, sliding_window * para.kernel[0], output_g_shape[2]};
    cout<< sliding_window * para.kernel[0];

    int W, H, C;
    W = output_g_shape[0];
    H = output_g_shape[1];
    C = output_g_shape[2];

    // /*------------------- MDSPAN SETUP -------------------*/
    // cobra::mdspan<int> in_l{in_l1_mem.data(), input_l_shape};
    cobra::mdspan<int> in_g{src.data(), input_g_shape};
    cobra::mdspan<int> out_g{dst.data(), output_g_shape};

    maxpool_cfunc(out_g.data, in_g.data, W, H, C, para.dilated_kernel[0],
                  para.stride[0], para.dilation[0], para.kernel[0], para);
    
    vector<int> trans_mem;
    vector<int> s = {1, 6, 9};
    
    strided<int>(0, 0, {54, 1, 6}, {1, 6, 9},trans_mem, out_g.data);
    cobra::mdspan<int> trans{trans_mem.data(), s};

    int out = (6 / para.kernel[1]) * para.stride[1] - 1+ para.dilated_kernel[1];
    cout <<"\n" << out << "\n";
    maxpool_cfunc(out_g.data, trans.data, 1, 6, 9, para.dilated_kernel[1],
                  para.stride[1], para.dilation[1], para.kernel[1], para);


    // print(&trans, 1);
    // (const vector<T1>& dst_st, const vector<T1>& src_st,
    //          const vector<T1>& dst_sh, const vector<T1>& src_sh,
    //          T2 *out, T2 *in, vector<int> dst_offset = {},
    //          vector<int> src_offset = {})

    // /******************************************************/
    // int w, h, c;
    // w = input_l_shape[0];
    // h = input_l_shape[1];
    // c = input_l_shape[2];

    // /*----- Initialize l1 shape and sub input offsets -----*/
    // int l1off_j, l1off_k;
    // int goff_j, goff_k;
    // int step_j = ALIGNDOWN((h - para.dilated_kernel[0]), para.stride[0]);
    // int step_k = ALIGNDOWN((c - para.dilated_kernel[1]), para.stride[1]);
    // int jump_j = step_j / para.stride[0] + 1;
    // int jump_k = step_k / para.stride[1] + 1;
    // h = step_j + para.dilated_kernel[0];
    // c = step_k + para.dilated_kernel[1];
    // step_j += para.stride[0];
    // step_k += para.stride[1];

    // /*---------- return shape corresponding to output ------------*/
    // get_out_shape(output_l_shape, w, h, c, para);
    // mdspan<int> out_l{out_l1_mem.data(), output_l_shape};

    // for (int i = 0; i < W; i += w) {
    //     for (int j = 0, j_out = 0; j < H; j += step_j, j_out += jump_j) {
    //         /*---------------- SETTING J(private and global) OFFSET TO HANDLE PADDING ----------*/
    //         set_in_offset_after_padding(l1off_j, goff_j, para.padding[0], j);

    //         for (int k = 0, k_out = 0; k < C; k += step_k, k_out += jump_k){
    //             /*------------ SETTING k(private and global) OFFSET TO HANDLE PADDING ----------*/
    //             set_in_offset_after_padding(l1off_k, goff_k, para.padding[1], k);

    //             //global -> private slicing
    //             cobra::mdspan<int> input{in_l1_mem.data(), {w, h, c}};
    //             cobra::slice(&input, &in_g, {0, l1off_j, l1off_k}, {i, goff_j, goff_k});

    //             /*-------------------- cfunc call ---------------------*/
    //             maxpool_cfunc(out_l.data, input.data, w, h, c, para);

    //             // private -> global slicing
    //             cobra::slice<int>(&out_g, &out_l, {i, j_out, k_out},
    //                                                {0, 0, 0});
    //         }
    //     }
    // }
    /******************************************************/
    print(&out_g, 1);
    print(&in_g, 1);
    return 0;
}
