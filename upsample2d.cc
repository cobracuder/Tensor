#include<bits/stdc++.h>
using namespace std;

template<typename T>
void init(vector<vector<T>> &d) {
    int count = 0;
    for (int i = 0; i < d.size(); i++) {
      for (int j = 0; j < d[0].size(); j++) {
        d[i][j] = ++count;
        cout<<count<<" ";
      }
      cout<<"\n";
    }
}

template<typename T>
void bilinear2D(vector<vector<T>> &out, vector<vector<T>> &in, bool align_corner = true) {
    int row_in = in.size();
    int col_in = in[0].size();
    int row_out = out.size();
    int col_out = out[0].size();
    float scale_factor_row, scale_factor_col;
    if (align_corner) {
        scale_factor_row = (row_in * 1.0f - 1) / (row_out - 1); // ek gap ko 2 gap pe krna hai
        scale_factor_col = (col_in * 1.0f - 1) / (col_out - 1); // ek gap ko 2 gap pe krna hai

        for (int i = 0; i < row_out; i++) {
        float idx_y = i * scale_factor_row;
        int low_y = floor(idx_y);
        int high_y = ceil(idx_y);
        high_y = min(high_y, row_in - 1);
        float weight_y = idx_y - low_y;
        for (int j = 0; j < col_out; j++) {
            float idx_x = j * scale_factor_col;
            int low_x = floor(idx_x);
            int high_x = ceil(idx_x);
            high_x = min(high_x, col_in - 1);
            float weight_x = idx_x - low_x;
            float val_1 = (in[low_y][low_x] * (1.0f - weight_x) + in[low_y][high_x] * (weight_x));
            float val_2 = (in[high_y][low_x] * (1.0f - weight_x) + in[high_y][high_x] * (weight_x));
            cout << (val_1 * (1.0f - weight_y) + val_2 * weight_y) << " ";
        }
        cout<<"\n";
        }
    } else {
        scale_factor_row = (row_in * 1.0f) / (row_out);
        scale_factor_col = (col_in * 1.0f) / (col_out);
        for (int i = 0; i < row_out; i++) {
            float idx_y = (i + 0.5f) * scale_factor_row - 0.5f;
            int low_y = max(floor(idx_y), 0.0f);
            int high_y = ceil(idx_y);
            high_y = min(high_y, row_in - 1);
            float weight_y = idx_y - low_y;
            for (int j = 0; j < col_out; j++) {
                float idx_x = (j + 0.5f) * scale_factor_col - 0.5f;
                int low_x = max(floor(idx_x), 0.0f);
                int high_x = ceil(idx_x);
                high_x = min(high_x, col_in - 1);
                float weight_x = idx_x - low_x;
                float val_1 = (in[low_y][low_x] * (1.0f - weight_x) + in[low_y][high_x] * (weight_x));
                float val_2 = (in[high_y][low_x] * (1.0f - weight_x) + in[high_y][high_x] * (weight_x));

                if (j == 0) {
                    cout << in[idx_y][j] <<" ";
                } else {
                    cout << (val_1 * (1.0f - weight_y) + val_2 * weight_y) << " ";
                }
            }
            cout<<"\n";
        }
    }
}

template<typename T>
auto return_vec(int row, int col) {
    vector<vector<T>> arr(row, vector<T>(col, 0));
    return arr;
}

int main() {
    auto arr = return_vec<float>(2, 2);
    auto out = return_vec<float>(4, 4);
    init(arr);
    cout<<"\n\n=============================================\n\n";
    bilinear2D(out, arr, 0);
}
