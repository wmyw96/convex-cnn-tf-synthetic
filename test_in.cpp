#include <iostream>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
using namespace std;

#define MAXN 2222
float a[MAXN][MAXN];
int vis[MAXN];
int perm[MAXN];
int ansperm[MAXN];
int seed_t;
int n;
float ansv;

void dfs(int cur){
    if (cur == n){
        float value = 0;
        for (int i = 0; i < n; ++i)
            value += a[i][perm[i]];
        if (value < ansv){
            ansv = value;
            for (int i = 0; i < n; ++i)
                ansperm[i] = perm[i];
        }
    }
    else{
        for (int i = 0; i < n; ++i)
            if (!vis[i]){
                perm[cur] = i;
                vis[i] = 1;
                dfs(cur + 1);
                vis[i] = 0;
            }
    }
}

int main(){
    ansv = 1e9;
    n = 1000;
    seed_t = 0;
    FILE* f1 = fopen("handle.txt", "r");
    fscanf(f1, "%d", &seed_t);
    fclose(f1);
    printf("SEED TICKED = %d\n", seed_t);
    srand(seed_t);

    FILE *f2 = fopen("handle.txt", "w");
    fprintf(f2, "%d\n", seed_t + 1);
    fclose(f2);
    
    FILE *f3 = fopen("mat.in", "w");

    for (int i = 0; i < n; ++i){
        for (int j = 0; j < n; ++j){
            a[i][j] = float(rand()) / 2147483647.0;
            a[i][j] = float(int(a[i][j] * 100000)) / 10000.0;
            fprintf(f3, "%.4f ", a[i][j]);
        }
        fprintf(f3, "\n");
    }
    fclose(f3);

    /*dfs(0);
    FILE *f4 = fopen("mat.ans", "w");
    fprintf(f4, "%.4f\n", ansv);
    for (int i = 0; i < n; ++i){
        fprintf(f4, i + 1 < n ? "%d " : "%d\n", ansperm[i]);
    }
    fclose(f4);*/
}