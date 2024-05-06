#include<bits/stdc++.h>
using namespace std;
vector<int> out,in;
void broadcast(int dim, int dst, int src, vector<int> dst_st, vector<int> src_st,
vector<int> dst_sh, vector<int> src_sh, int rank){
    if(dim == rank){
        out[dst] = in[src];
        // cout<<"\n hello "<<in[src + src_st[dim] * (src_sh[dim]==1?0:i)]<<" "<<src + src_st[dim] * (src_sh[dim]==1?1:i)<<endl;
        return;
    }
    for(int i = 0; i < dst_sh[dim]; i++){
        broadcast(dim + 1, (dst + dst_st[dim] * i), 
                  (src + src_st[dim] * (src_sh[dim]==1?0:i)),
                  dst_st, src_st, dst_sh, src_sh, rank);
    }
}
 int main(){
    out.resize(16);
    vector<int> dst_sh = {2, 2, 2, 2};
    vector<int> dst_st = {8, 4, 2, 1};
    in = {2, 1};
    vector<int> src_sh = {1, 1, 1, 2};
    vector<int> src_st = {2, 2, 2, 1};
    broadcast(0, 0, 0, dst_st, src_st, dst_sh, src_sh, dst_sh.size());
    for (auto i :out){
        cout<<i<<" ";
    }
 }
