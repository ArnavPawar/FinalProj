all: main.c main.cu
	mpixlc -O3 main.c -c -o main-xlc.o
	nvcc -O3 -arch=sm_70 main.cu -c -o main-nvcc.o 
	mpixlc -O3 main-xlc.o main-nvcc.o -o main-exe -L/usr/local/cuda-11.2/lib64/ -lcudadevrt -lcudart -lstdc++ 