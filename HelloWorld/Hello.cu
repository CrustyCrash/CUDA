#include <stdio.h>

__global__ void kernal ( void ){

}

int main(){
    kernal <<<1,1>>> ();
    printf("Hello World!");
    
}