#include <stdio.h>
#define N 10

void add(int* a, int* b, int* c)
{
    int tid = 0; //can be changed depending on number of hosts
    while( tid < 10)
    {
        c[tid] = a[tid] + b[tid];
        tid++;
    }
}

int main()
{
    int a[N];
    int b[N];
    int c[N];

    //populating the arrays
    for(int i = 0; i < N; i++)
    {
        a[i] = -i;
        b[i] = i * i;
    }
    add(a,b,c);

    for(int i = 0; i < N; i++)
    {
        printf( "%d + %d = %d\n", a[i], b[i], c[i] );
    }
}
