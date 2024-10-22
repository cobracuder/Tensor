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
    float scale_ratio_d;
    float scale_ratio_h;
    float scale_ratio_w;
};

int64_t linear_idx(int64_t i, int64_t j, int64_t k, int64_t l,
                              int64_t d, int64_t h, int64_t w, int64_t nc) {
  return (i * (h * w * nc) + j * (w * nc) + k * (nc) + l);
}


/* PARAM DETAILS
 * out: l1 out buffer pointer
 * input: l1 in buffer pointer
 * d: private out depth size
 * h: private out height size
 * w: private out width size
 * nc: private out (batch X channel) size
 * out_g_idx_d: global out d dim idx
 * out_g_idx_h: global out h dim idx
 * out_g_idx_w: global out w dim idx
 * in_g_idx_d: global in d dim idx
 * in_g_idx_h: global in h dim idx
 * in_g_idx_w: global in w dim idx
 * d_in: private in depth size
 * h_in: private in height size
 * w_in: private in width size
 * scale_ratio_d: scale ratio for dim d (1.0 / scale_factor_d)
 * scale_ratio_h: scale ratio for dim h (1.0 / scale_factor_h)
 * scale_ratio_w: scale ratio for dim w (1.0 / scale_factor_w)
 */
template <typename T>
void upsample_trilinear_cfunc_align_true(
    T *out, T *input, int d, int h, int w, int nc, int out_g_idx_d,
    int out_g_idx_h, int out_g_idx_w, int in_g_idx_d, int in_g_idx_h,
    int in_g_idx_w, int d_in, int h_in, int w_in, float scale_ratio_d,
    float scale_ratio_h, float scale_ratio_w) {


  float vec_000, vec_001, vec_010, vec_011, vec_100, vec_101, vec_110, vec_111;
  float v_one = 1.0f;

  for (int i = 0; i < d; i++) {
    // input d dim details
    float tmp_in_idx_d = (i + out_g_idx_d) * scale_ratio_d;
    int low_d = floor(tmp_in_idx_d);
    float weight_d = tmp_in_idx_d - low_d;
    low_d -= in_g_idx_d;
    int high_d = std::min((ceilf(tmp_in_idx_d) - in_g_idx_d), d_in * 1.0f);

    for (int j = 0; j < h; j++) {
      // input h dim details
      float tmp_in_idx_h = (j + out_g_idx_h) * scale_ratio_h;
      int low_h = floorf(tmp_in_idx_h);
      float weight_h = tmp_in_idx_h - low_h;
      low_h -= in_g_idx_h;
      int high_h = std::min((ceilf(tmp_in_idx_h) - in_g_idx_h), h_in * 1.0f);

      for (int k = 0; k < w; k++) {
        // input w dim details
        float tmp_in_idx_w = (k + out_g_idx_w) * scale_ratio_w;
        int low_w = floorf(tmp_in_idx_w);
        float weight_w = tmp_in_idx_w - low_w;
        low_w -= in_g_idx_w;
        int high_w = std::min((ceilf(tmp_in_idx_w) - in_g_idx_w), w_in * 1.0f);
        for (int l = 0; l < nc; l += 1) {

          /*---------------- LOADING INPUT 8 NEAREST POINTS -----------------*/
          vec_000 = input[linear_idx(low_d, low_h, low_w, l, d_in, h_in, w_in, nc)];
          vec_001 = input[linear_idx(low_d, low_h, high_w, l, d_in, h_in, w_in, nc)];
          vec_010 = input[linear_idx(low_d, high_h, low_w, l, d_in, h_in, w_in, nc)];
          vec_011 = input[linear_idx(low_d, high_h, high_w, l, d_in, h_in, w_in, nc)];
          vec_100 = input[linear_idx(high_d, low_h, low_w, l, d_in, h_in, w_in, nc)];
          vec_101 = input[linear_idx(high_d, low_h, high_w, l, d_in, h_in, w_in, nc)];
          vec_110 = input[linear_idx(high_d, high_h, low_w, l, d_in, h_in, w_in, nc)];
          vec_111 = input[linear_idx(high_d, high_h, high_w, l, d_in, h_in, w_in, nc)];
        //   printf("\n%f, %f, %f, %f, %f, %f, %f, %f\n", vec_000, vec_001, vec_010, vec_011, vec_100, vec_101, vec_110, vec_111);

          /*------------ Processing these point dimension wise -------------*/
          // D0 dim
          float d0_h0_mid = vec_000 * (v_one - weight_w) + vec_001 * weight_w;
          float d0_h1_mid = vec_010 * (v_one - weight_w) + vec_011 * weight_w;
          float d0_mid =
              d0_h0_mid * (v_one - weight_h) + d0_h1_mid * (weight_h);

          // D1 dim
          float d1_h0_mid = vec_100 * (v_one - weight_w) + vec_101 * weight_w;
          float d1_h1_mid = vec_110 * (v_one - weight_w) + vec_111 * weight_w;
          float d1_mid =
              d1_h0_mid * (v_one - weight_h) + d1_h1_mid * (weight_h);

          // final output
          float v_res = d0_mid * (v_one - weight_d) + d1_mid * weight_d;
          out[linear_idx(i, j, k, l, d, h, w, nc)] = v_res;
        }
      }
    }
  }
} 

double scale_fac(int in_dim, int out_dim, op_para para) {
    if (para.align_corner) {
        return ((in_dim * 1.0f - 1) / max(out_dim - 1, 1));
    } else {
        return ((in_dim * 1.0f) / max(out_dim, 1));
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

    para.align_corner = 2;
    /*------------------- SHAPES -------------------------*/
    vector<int> input_g_shape = {1, 2, 2, 4};
    vector<int> input_l_shape = {1, 3, 3, 4};
    vector<int> output_g_shape = {1, 2, 2, 2};
    vector<int> output_l_shape = {1, 3, 3, 4};

    /*-------------------- MDSPAN -----------------------*/
    mdspan<float> src_g{src.data(), input_g_shape};
    mdspan<float> dst_g{dst.data(), output_g_shape};

    
    int *in_ptr = input_g_shape.data();
    int *ptr = output_g_shape.data();
    int *out = output_l_shape.data();

    if (para.align_corner) {
        para.scale_ratio_d = scale_fac(in_ptr[1], output_g_shape[1], para); // ek gap ko 2 gap pe krna hai
        para.scale_ratio_h = scale_fac(in_ptr[2], output_g_shape[2], para); // ek gap ko 2 gap pe krna hai
        para.scale_ratio_w = scale_fac(in_ptr[3], output_g_shape[3], para); // ek gap ko 2 gap pe krna hai
        cout<<para.scale_ratio_d <<" "<<para.scale_ratio_h<<" "<<para.scale_ratio_w<<endl;
        for (int i = 0; i < ptr[0]; i+= out[0]) {
            int rem_n = ptr[0] - i >= out[0] ? out[0]: ptr[0] - i;

            for (int j = 0; j < ptr[1]; j += out[1]) {
                int rem_d = ptr[1] - j >= out[1] ? out[1]: ptr[1] - j;
                float low_idx_d = j * para.scale_ratio_d;
                float high_idx_d = (j + rem_d - 1) * para.scale_ratio_d;
                int low_d = max(0.0f, floor(low_idx_d));
                int high_d = ceil(high_idx_d);
                high_d = min(high_d, in_ptr[1] - 1);

                for (int k = 0; k < ptr[2]; k += out[2]) {
                    int rem_h = ptr[2] - k >= out[2] ? out[2]: ptr[2] - k;
                    float low_idx_h = k * para.scale_ratio_h;
                    float high_idx_h = (k + rem_h - 1) * para.scale_ratio_h;
                    int low_h = max(0.0f, floor(low_idx_h));
                    int high_h = ceil(high_idx_h);
                    high_h = min(high_h, in_ptr[2] - 1);

                    for (int l = 0; l < ptr[3]; l += out[3]) {
                        int rem_w = ptr[3] - l >= out[3] ? out[3]: ptr[3] - l;
                        float low_idx_w = l * para.scale_ratio_w;
                        float high_idx_w = (l + rem_w - 1) * para.scale_ratio_w;
                        int low_w = max(0.0f, floor(low_idx_w));
                        int high_w = ceil(high_idx_w);
                        high_w = min(high_w, in_ptr[3] - 1);

                        cobra::mdspan<float>in_l(in_l1_mem.data(),
                                                 {rem_n, (high_d - low_d + 1),
                                                  (high_h - low_h + 1), (high_w - low_w + 1)});
                        cobra::slice(&in_l, &src_g, {0, 0, 0, 0}, {i, low_d, low_h, low_w});
                        print(&in_l);

                        upsample_trilinear_cfunc_align_true(out_l1_mem.data(), in_l.data, rem_d, rem_h, rem_w, rem_n, j, k, l,
                                                            low_d, low_h, low_w, (high_d - low_d + 1),
                                                            (high_h - low_h + 1), (high_w - low_w + 1),
                                                            para.scale_ratio_d, para.scale_ratio_h, para.scale_ratio_w);
                        cobra::mdspan<float>out_l(out_l1_mem.data(), {rem_n, rem_d, rem_h, rem_w});

                        print(&out_l);
                        cobra::slice(&dst_g, &out_l, {i, j, k, l}, {0, 0, 0, 0});

                    }

                }
            }
        }
    }

    print(&src_g);
    print(&dst_g);
    return 0;
}
