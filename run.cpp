#include <iostream>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
using namespace std;

int main(){
    bool ck = true;
    int testcase = 0;
    while (ck){
        printf("======== Test %d ========\n", testcase++);
        system("./test");
        system("./hungarian");
        ck = system("diff mat.out mat.ans");
        ck = !ck;
        if (ck){
            printf("No difference found...\n");
        }
    }
}