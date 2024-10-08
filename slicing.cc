#include<bits/stdc++.h>
using namespace std;
#define DEBUG_OPTION 1

/*----------- mdsan struct ---------*/
template<typename T>
struct md_span{
    T *data;
    vector<int> shape;
};

/*------- To initialize data ------*/
template<typename T>
void initialize(vector<T>&data) {
    int size = data.size();
    for(int i = 0; i < size; i++) {
        data[i] = i + 1;
    }
} // initialize

/*------- To Calc Stride -------*/
template<typename T>
auto cal_stride(vector<T>shape) {
    vector<T>stride(shape.size(), 1);
    for (int64_t i = shape.size() - 2; i >= 0; --i) {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
    return stride;
} // cal_stride

/*-------- print function ---------*/
template<typename T>
void print(vector<T>arr) {
    if (DEBUG_OPTION) {
        std::cout<<endl;
        for (auto i: arr) {
            std::cout<<i<<", ";
        }
        std::cout<<endl;
    }
} //print

template<typename T>
void print(T s) {
    if (DEBUG_OPTION) {
        std::cout<<endl;
        std::cout<<s;
        std::cout<<endl;
    }
} //print

/*----- To Print Multi-Dimension array ------*/
template<typename T>
void dis(int dim, int idx, T *data, vector<int>st, vector<int>sh, int rank) {
    if (dim == rank) {
        std::cout<<data[idx]<<", ";
        return;
    }
    
    std::cout<<"[ ";
    for (int i = 0; i < sh[dim]; i++) {
        dis(dim + 1, idx + st[dim] * i, data, st, sh, rank);
    }
    std::cout<<"]\n";

} //dis

template<typename T>
void print(md_span<T> *ob) {
    if (DEBUG_OPTION) {
        std::cout<<endl;
        std::cout<<"Shape: \n";
        for (auto i: ob->shape) {
            std::cout<<i<<", ";
        }
        std::cout<<"\nDATA: \n";
        dis(0, 0, ob->data, cal_stride(ob->shape), ob->shape, ob->shape.size());
        std::cout<<endl;
    }
}  // print

/*--------- cpy function ------------*/
template<typename T1, typename T2>
void cpy(int dim, int dst, int src, vector<T1> dst_st, vector<T1> src_st,
         vector<T1> dst_sh, vector<T1> src_sh, int rank, T2 *out, T2 *in,
         vector<int>src_offset, vector<int>dst_offset){
    if(dim == rank){
        out[dst] = in[src];
        return;
    }
    for(int i = dst_offset[dim], j = src_offset[dim];
        i < dst_sh[dim] && j < src_sh[dim]; i++, j++){
        cpy(dim + 1, (dst + dst_st[dim] * i), 
            (src + src_st[dim] * (j)),
            dst_st, src_st, dst_sh, src_sh, rank, out, in, src_offset, dst_offset);
    }
} //cpy

/*--------- slice function ----------*/
template<typename T>
void sliced(md_span<T> *des, md_span<T> *source,
            vector<int> dst_offset = {},
            vector<int> src_offset = {}) {
    auto des_st = cal_stride(des->shape);
    auto src_st = cal_stride(source->shape);
    int rank = des_st.size();

    if (dst_offset.size() == 0) {
        dst_offset.resize(rank, 0);
    }
    if (src_offset.size() == 0) {
        src_offset.resize(rank, 0);
    }

    cpy(0, 0, 0, des_st, src_st, des->shape, source->shape, rank,
        des->data, source->data, src_offset, dst_offset);
} //sliced

int main() {
    vector<int>arr(27,0);
    vector<int>src(125);
    initialize(src);
    md_span<int> *ob1 = new md_span<int>{arr.data(), {3, 3, 3}};
    md_span<int> *ob2 = new md_span<int>{src.data(), {5, 5, 5}};
    sliced<int>(ob1, ob2);
    print(ob2);
    print(ob1);
}
