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
void print(const std::vector<T> arr, bool flag = 0) {
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
void strided(int dim, int index, vector<int>strides, vector<int>shape, vector<T>&dst, T *src) {
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
    para.stride[0] = 1;
    para.stride[1] = 1;
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
    vector<int> in_l1_mem(1000, 0);
    vector<int> src(125000);
    vector<int> out_l1_mem(1000, 0); // INT_MIN is used to track uninitialized areas
    vector<int> dst(12500);
    vector<int> intermediate(12500, 0);

    initialize(src);  // Initialize input with sequential values

    /*------------------ INITIALIZE PARAMETERS ----------------*/
    op_para para;
    set_op_para(para);

    /*------------------- SHAPES -------------------------*/
    vector<int> output_g_shape = {1, 4, 4};

    int sliding_window = ceil((output_g_shape[1] - para.dilated_kernel[0] + 1.0f) / para.stride[0]);
    int sliding_window2 = ceil((output_g_shape[2] - para.dilated_kernel[1] + 1.0f) / para.stride[1]);
    vector<int> input_g_shape = {1, sliding_window * para.kernel[0], sliding_window2 * para.kernel[1]};
    vector<int> inter_shape = {1, output_g_shape[1], input_g_shape[2]};

    int W, H, C;
    W = input_g_shape[0];
    H = input_g_shape[1];
    C = input_g_shape[2];
    int h = para.kernel[0] * 2, c = 128;

    cobra::mdspan<int> in_g{src.data(), input_g_shape};
    cobra::mdspan<int> out_g{dst.data(), output_g_shape};
    cobra::mdspan<int> inter{intermediate.data(), inter_shape};

    int pre = 0;

    print(vector<int>{H, C}, 1);
    for (int i = 0; i < H; i += h) {
        int rem = H - i > h? h: H - i;
        int out_h = (rem / para.kernel[0] - 1) * para.stride[0] + para.dilated_kernel[0];
        cobra::mdspan<int> in_l{in_l1_mem.data(), {1, rem, c}};
        cobra::mdspan<int> out_l{out_l1_mem.data(), {1, out_h, c}};
        int offset_h_out = (i / para.kernel[0]) * para.stride[0];
        for (int j = 0; j < C; j += c) {
            cobra::slice<int>(&in_l, &in_g, {0, 0, 0}, {0, i, j});
            cobra::slice<int>(&out_l, &inter, {0, 0, 0}, {0, offset_h_out, j});
            maxpool_cfunc(out_l.data, in_l.data, 1, out_h, c,
                  para.dilated_kernel[0],
                  para.stride[0], para.dilation[0], para.kernel[0], para);
            cobra::slice<int>(&inter, &out_l, {0, offset_h_out, j});
        }
    }

    h = 128, c = para.kernel[1] * 2;
    H = output_g_shape[1];

    // sync thread

    for (int i = 0; i < C; i += c) {
        int rem = C - i > c? c: C - i;
        int out_c = (rem / para.kernel[1] - 1) * para.stride[1] + para.dilated_kernel[1];
        cobra::mdspan<int> in_l{in_l1_mem.data(), {1, h, rem}};
        cobra::mdspan<int> out_l{out_l1_mem.data(), {1, h, out_c}};
        int offset_c_out = (i / para.kernel[1]) * para.stride[1];
        for (int j = 0; j < H; j += h) {
            // intermediate load and transpose
            cobra::slice<int>(&in_l, &inter, {0, 0, 0}, {0, j, i});
            vector<int>tmp_in;
            strided<int>(0, 0, {h * rem, 1, rem}, {1, rem, h}, tmp_in, in_l.data);

            // final out load and transpose
            cobra::slice<int>(&out_l, &out_g, {0, 0, 0}, {0, j, offset_c_out});
            vector<int>tmp_out;
            strided<int>(0, 0, {h * out_c, 1, out_c}, {1, out_c, h}, tmp_out, out_l.data);

            // cfunc call
            maxpool_cfunc(tmp_out.data(), tmp_in.data(), 1, out_c, h,
                  para.dilated_kernel[1],
                  para.stride[1], para.dilation[1], para.kernel[1], para);
            
            // tranpose and store output
            vector<int>out;
            strided<int>(0, 0, {h * out_c, 1, h}, {1, h, out_c}, out, tmp_out.data());
            cobra::mdspan<int> output{out.data(), {1, h, out_c}};
            cobra::slice<int>(&out_g, &output, {0, j, offset_c_out});
        }
    }

    print(&in_g, 1);
    print(&inter, 1);
    print(&out_g, 1);
    /******************************************************/
    return 0;
}
