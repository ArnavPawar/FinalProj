#include "main.h"
#include<cuda.h>
#include<cuda_runtime.h>

extern "C" 
{
    void addToList(struct Node* n, struct List *li, int val, int rank);
    void deleteNode(struct List *li, struct Node* n, int rank);
    const char* getSeasonName(int val);
}

const char* getSeasonName(int val) {
    switch (val) {
        case 0: return "Spring";
        case 1: return "Summer";
        case 2: return "Autumn";
        case 3: return "Winter";
        default: return "Unknown"; // Fallback case
    }
}

// CUDA kernel to add two numbers
__global__ void addToListCuda(struct Node* n, int val) {
    n->id = val;
    n->next = NULL;
}

__global__ void deleteNodeCuda(struct Node* n, struct List *li) {
    if (n->next != NULL && n->prev != NULL) { // middle case (node is in middle of list)
        n->next->prev = n->prev;
        n->prev->next = n->next;
    } else { // end case (node is head or end of list)
        if (n->next != NULL) { // node is head of list
            n->next->prev = NULL;
            li->head = n->next;
        } else if (n->prev != NULL) { // node is end of list
            n->prev->next = NULL;
            li->end = n->prev;
        } else { // node is head and end of list (prev and next == NULL)
            li->head = NULL;
            li->end = NULL;
        }
    }
    n->next = NULL;
    n->prev = NULL;
}

void addToList(struct Node* n, struct List *li, int val, int rank) {
    cudaSetDevice( rank % 4 );
    struct Node *n_temp;
    cudaMalloc((void**)&n_temp, sizeof(struct Node));
    cudaMemcpy(n_temp, n, sizeof(struct Node), cudaMemcpyHostToDevice);
    addToListCuda<<<128, 128>>>(n_temp, val);
    cudaMemcpy(n, n_temp, sizeof(struct Node), cudaMemcpyDeviceToHost);
    cudaFree(n_temp);
    if (li->size == 0) {
        n->prev = NULL;
        li->head = n;
    } else {
        n->prev = li->end;
        li->end->next = n;
    }
    li->end = n;
    li->size += 1;
}

void deleteNode(struct List *li, struct Node* n, int rank) {
    cudaSetDevice( rank % 4 );
    struct Node *n_temp;
    struct List *li_temp;
    cudaMalloc((void**)&n_temp, sizeof(struct Node));
    cudaMalloc((void**)&li_temp, sizeof(struct List));
    cudaMemcpy(n_temp, n, sizeof(struct Node), cudaMemcpyHostToDevice);
    cudaMemcpy(li_temp, li, sizeof(struct List), cudaMemcpyHostToDevice);
    deleteNodeCuda<<<128, 128>>>(n_temp, li_temp);
    cudaMemcpy(n, n_temp, sizeof(struct Node), cudaMemcpyDeviceToHost);
    cudaMemcpy(li, li_temp, sizeof(struct List), cudaMemcpyDeviceToHost);
    cudaFree(n_temp);
    cudaFree(li_temp);
    cudaFree(n);
    li->size -= 1;
}