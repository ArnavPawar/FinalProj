#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#define num_trees 256
#define connections 5
#define spread_chance 0.3
#define extinguish_chance 0.1
#define startBurningChance 0.05

struct Cell {
    int status;  // 0 = empty, 1 = tree, 2 = burning
    int neighbors[connections];
};

__global__ void setupRNG(curandState *state, unsigned long seed) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(seed, id, 0, &state[id]);
}

__global__ void initForest(Cell *forest, curandState *states) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id < num_trees) {
        curandState localState = states[id];
        forest[id].status = (curand_uniform(&localState) < startBurningChance) ? 2 : 1;
        for (int i = 0; i < connections; i++) {
            forest[id].neighbors[i] = curand(&localState) % num_trees;
        }
        states[id] = localState;
    }
}

__global__ void spreadFire(Cell *forest, curandState *states, int *changes) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id < num_trees && forest[id].status == 2) {
        curandState localState = states[id];
        for (int i = 0; i < connections; i++) {
            int nid = forest[id].neighbors[i];
            if (forest[nid].status == 1 && curand_uniform(&localState) < spread_chance) {
                atomicExch(&forest[nid].status, 2);
                atomicAdd(&changes[0], 1);
            }
        }
        if (curand_uniform(&localState) < extinguish_chance) {
            forest[id].status = 0;
            atomicAdd(&changes[1], 1);
        }
        states[id] = localState;
    }
}

int main() {
    Cell *d_forest;
    curandState *d_states;
    int *d_changes;
    int changes[2] = {0, 0};

    cudaMalloc(&d_forest, num_trees * sizeof(Cell));
    cudaMalloc(&d_states, num_trees * sizeof(curandState));
    cudaMalloc(&d_changes, 2 * sizeof(int));
    cudaMemcpy(d_changes, changes, 2 * sizeof(int), cudaMemcpyHostToDevice);

    dim3 blocks((num_trees + 255) / 256);
    dim3 threads(256);

    setupRNG<<<blocks, threads>>>(d_states, time(NULL));
    initForest<<<blocks, threads>>>(d_forest, d_states);
    cudaDeviceSynchronize();

    for (int day = 0; day < 100; day++) {
        cudaMemcpy(d_changes, changes, 2 * sizeof(int), cudaMemcpyHostToDevice);
        spreadFire<<<blocks, threads>>>(d_forest, d_states, d_changes);
        cudaDeviceSynchronize();
        cudaMemcpy(changes, d_changes, 2 * sizeof(int), cudaMemcpyDeviceToHost);
        if (changes[0] == 0 && changes[1] == 0) break; // No new fires and no new extinguishments
        printf("Day %d: New Fires: %d, Extinguished: %d\n", day, changes[0], changes[1]);
        changes[0] = 0;
        changes[1] = 0;
    }

    cudaFree(d_forest);
    cudaFree(d_states);
    cudaFree(d_changes);
    return 0;
}
