#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include <stdbool.h>
#include "clockcycle.h"

#define num_trees 20000000
#define neighbors 5
#define burn_chance 0.3
#define die_chance 0.8
#define clock_frequency 512000000
#define start_burn_chance 0.1

struct Cell
{
    long long int id;
    int status;
    long long int neighbors[num_neighbors];
};

struct List {
    struct Node* head;
    struct Node* end;
    long long int size;
};

struct Node {
    long long int id;
    struct Node* next;
    struct Node* prev;
};

struct Node* addToList(struct List *li, long long int val) {
    struct Node* n = malloc(sizeof(struct Node));
    n->id = val;
    n->next = NULL;
    if (li->size == 0) {
        n->prev = NULL;
        li->head = n;
    } else {
        n->prev = li->end;
        li->end->next = n;
    }

    li->end = n;
    li->size += 1;
    return n;
}

void deleteNode(struct List *li, struct Node* n) {
    if (n->next != NULL && n->prev != NULL) {
        n->next->prev = n->prev;
        n->prev->next = n->next;
    } else {
        if (n->next != NULL) {
            n->next->prev = NULL;
            li->head = n->next;
        } else if (n->prev != NULL) {
            n->prev->next = NULL;
            li->end = n->prev;
        } else {
            li->head = NULL;
            li->end = NULL;
        }
    }
    n->next = NULL;
    n->prev = NULL;
    free(n);
    li->size -= 1;
}

void start_simulation(int myrank, int numranks) {
    struct Cell *tree = calloc(num_trees, sizeof(struct Cell));
    struct List burn_list = {NULL, NULL, 0};
    struct List burn_queue = {NULL, NULL, 0};
    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, "output.txt", MPI_MODE_CREATE | MPI_MODE_WRONLY,
            MPI_INFO_NULL, &file);
    
    srand(myrank);
    long long int id = myrank;
    while (id < num_trees) {
        long long int neighbors[num_neighbors];
        for (int c = 0; c < num_neighbors; c++) {
            neighbors[c] = rand() % num_trees;
        }

        float chance = (double)rand() / RAND_MAX;
        int is_burn = 0;
        if (chance < start_burn_chance) {
            addToList(&burn_list, id);
            is_burn = 1;
        }
        struct Cell p = {id, is_burn};
        memcpy(p.neighbors, neighbors, num_neighbors * sizeof(long long int));
        tree[id] = p;

        id+= numranks;
    }
   
    int b = 0;
    while (true) {
        b += 1;
        long long int dayData[3] = {(num_trees/numranks)-burn_list.size-dead_count, burn_list.size, dead_count};
        long long int dayTotals[3];
        MPI_Allreduce(&dayData, &dayTotals, 3, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);

        if (dayTotals[0] == 0 || dayTotals[1] == 0) {

            char buffer[150];
            memset(buffer, '\0', sizeof(buffer)); 
            sprintf(buffer, "End of Sim Day %i Rank %i: %lli tree(s) not infected, %lli tree(s) infected, %lli tree(s) dead\n", b, myrank, dayData[0], dayData[1], dayData[2]);
            int count = strlen(buffer);
            MPI_Offset file_size;
            MPI_File_get_size(file, &file_size);

            MPI_Offset offset = file_size + (count * myrank);
            MPI_File_write_at_all(file, offset, buffer, count, MPI_CHAR, MPI_STATUS_IGNORE);
            MPI_File_sync(file);

            if (myrank == 0) {
                char buffer0[150];
                memset(buffer0, '\0', sizeof(buffer)); 
                sprintf(buffer0, "End of Sim Day %i Totals: %lli tree(s) not infected, %lli tree(s) infected, %lli tree(s) dead\n", b, dayTotals[0], dayTotals[1], dayTotals[2]);
                int count = strlen(buffer0);
                MPI_Offset file_size;
                MPI_File_get_size(file, &file_size);

                MPI_Offset offset = file_size;
                MPI_File_write_at(file, offset, buffer0, count, MPI_CHAR, MPI_STATUS_IGNORE);
            }
            break;
        } else {
            char buffer[150];
            memset(buffer, '\0', sizeof(buffer)); 
            sprintf(buffer, "Day %i Rank %i: %lli tree(s) not infected, %lli tree(s) infected, %lli tree(s) dead\n", b, myrank, dayData[0], dayData[1], dayData[2]);
            int count = strlen(buffer);
            MPI_Offset file_size;
            MPI_File_get_size(file, &file_size);

            MPI_Offset offset = file_size + (count * myrank);
            MPI_File_write_at_all(file, offset, buffer, count, MPI_CHAR, MPI_STATUS_IGNORE);
            MPI_File_sync(file);

            if (myrank == 0) {
                char buffer0[150];
                memset(buffer0, '\0', sizeof(buffer)); 
                sprintf(buffer0, "Day %i Totals: %lli tree(s) not infected, %lli tree(s) infected, %lli tree(s) dead\n", b, dayTotals[0], dayTotals[1], dayTotals[2]);
                int count = strlen(buffer0);
                MPI_Offset file_size;
                MPI_File_get_size(file, &file_size);
                MPI_Offset offset = file_size;
                MPI_File_write_at(file, offset, buffer0, count, MPI_CHAR, MPI_STATUS_IGNORE);
            }
        }

        struct Node* n = burn_list.head;
        long long int messageCount[numranks];
        memset( messageCount, 0, numranks*sizeof(long long int) );
        while (n != NULL) { 
            long long int id = n->id;

            for (int c = 0; c < num_neighbors; c++) {
                float chance = (double)rand()/RAND_MAX;
                if (chance < burn_chance) {
                    long long int infectedID = tree[id].neighbors[c];
                    if (infectedID % numranks == myrank) {
                        if (tree[infectedID].status == 0) {
                            addToList(&burn_queue, infectedID);
                            tree[infectedID].status = 1;
                        }
                    } else {
                        messageCount[infectedID % numranks] += 1;
                        MPI_Request request;
                        MPI_Isend(&infectedID, 1, MPI_LONG_LONG_INT, infectedID % numranks, 0, MPI_COMM_WORLD, &request);
                    }
                }
            }

            bool dead = false;
            float chance = (double)rand()/RAND_MAX;
           
            if (chance < die_chance) {
                tree[id].status = 2;          
                struct Node* temp = n->next;
                deleteNode(&burn_list, n);
                dead_count += 1;
                n = temp;
                dead = true;
            }
            if (!dead) {
                n = n->next;
            }
        }

        long long int totalMessageCount[numranks];
        MPI_Allreduce(&messageCount, &totalMessageCount, numranks, MPI_LONG_LONG_INT, MPI_SUM,  MPI_COMM_WORLD);

        for (long long int i = 0; i < totalMessageCount[myrank]; i++) {
            long long int recv;
            MPI_Recv(&recv, 1, MPI_LONG_LONG_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (tree[recv].status == 0) {
                tree[recv].status = 1;
                addToList(&burn_queue, recv);
                
            }
        }

        n = burn_queue.head;
        while(n != NULL) {
            long long int id = n->id;
            addToList(&burn_list, id);
            deleteNode(&burn_queue, n);
            n = burn_queue.head;
        }
    }
    MPI_File_close(&file);

}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int myrank;
    int numranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);            
    double start_cycles = clock_now();
    start_simulation(myrank, numranks);
    double end_cycles= clock_now();
    double secs = ((double)(end_cycles - start_cycles)) / clock_frequency;
    if (myrank == 0) {
        printf("CPU Reduce time: %f\n", secs);
    }
    MPI_Finalize();
    return 0;
}