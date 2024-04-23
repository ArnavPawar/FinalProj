#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <mpi.h>
#include <stdbool.h>
#include "clockcycle.h"

#define num_trees 256
#define connections 5
#define spread_chance 0.3
#define extinguish_chance 0.1
#define clock_frequency 512000000
#define startBurningChance 0.05
#define rainProb .01
#define sunnyProb .5
#define snowProb .03
#define cloudProb .2


struct Cell {
    int id;
    int status; // 1 = tree, 0 = empty, 2 = burning
    int neighbors[connections];
};

struct List {
    struct Node* head;
    struct Node* end;
    int size;
};

struct Node {
    int id;
    struct Node* next;
    struct Node* prev;
};

// adds node with id = val to end of list
struct Node* addToList(struct List *li, int val) {
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
// removes node from list and frees memory
void deleteNode(struct List *li, struct Node* n) {
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
    free(n);
    li->size -= 1;
}

void sim(int myrank, int numranks) {
    // initialize rank vars
    struct Cell *land = calloc(num_trees, sizeof(struct Cell));
    struct List infectedList = {NULL, NULL, 0}; // linked list for easy add/remove
    struct List infectedQueue = {NULL, NULL, 0};
    int burntCount = 0;
    MPI_File file;
    MPI_File_open(MPI_COMM_WORLD, "output.txt", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
    // initialize structs
    srand(myrank);
    int id = myrank; // the id of the Cell we're creating

    //CAHNGE!!!!
    //make a random function call that will decide what season it is outside
    while (id < num_trees) {
        // generate list of connections
        int neighbors[connections];
        for (int c = 0; c < connections; c++) {
            neighbors[c] = rand() % num_trees; // rand() % x = random number from 0 to x-1 ( e.g. [0-X) )
        }
        // determine if node starts infected
        float chance = (double)rand()/RAND_MAX;
        int isInfected = 0;

        //CHANGE!!!!
        // make this change the chance of starting burning based on the weather outside
        if (chance < startBurningChance) {
            addToList(&infectedList, id);
            isInfected = 1;
            //printf("Rank %i: Node %i has been infected at start\n", myrank, id); // Debug print
        }
        // generate struct and copy in list of connections
        struct Cell p = {id, isInfected};
        memcpy(p.neighbors, neighbors, connections * sizeof(int));
        land[id] = p;

        id+= numranks;
    }
   
    // day cycle
    int b = 0;
    while (true) {
        // Beginning of day information
        b += 1;
        int dayData[3] = {(num_trees/numranks)-infectedList.size-burntCount, infectedList.size, burntCount};
        int dayTotals[3];
        MPI_Allreduce(&dayData, &dayTotals, 3, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
        // Day summaries
        if (dayTotals[0] == 0 || dayTotals[1] == 0) { // sim ends when either no infected or non-infected cell remain

            // Print final individual rank day summary
            char buffer[150]; //buffer needed for io
            memset(buffer, '\0', sizeof(buffer)); 
            // sprintf(buffer, "End of Sim Day %i Rank %i: %i places not on fire, %i places on fire, %i burnt locations\n", b, myrank, dayData[0], dayData[1], dayData[2]);
            int count = strlen(buffer); // exclude null terminator
            // Determine the size of the file
            MPI_Offset file_size;
            MPI_File_get_size(file, &file_size);

            // Set the offset to the end of the file
            MPI_Offset offset = file_size + (count * myrank);
            MPI_File_write_at_all(file, offset, buffer, count, MPI_CHAR, MPI_STATUS_IGNORE);
            MPI_File_sync(file);

            // Print final total day summary
            if (myrank == 0) {
                char buffer0[150]; //buffer needed for io
                memset(buffer0, '\0', sizeof(buffer)); 
                sprintf(buffer0, "End of Sim Day %i Totals: %i places not on fire, %i places on fire, %i burnt locations\n", b, dayTotals[0], dayTotals[1], dayTotals[2]);
                int count = strlen(buffer0); // exclude null terminator
                // Determine the size of the file
                MPI_Offset file_size;
                MPI_File_get_size(file, &file_size);

                // Set the offset to the end of the file
                MPI_Offset offset = file_size;
                MPI_File_write_at(file, offset, buffer0, count, MPI_CHAR, MPI_STATUS_IGNORE);
            }
            break;
        } 
        else {
            // Print individual rank day summary
            char buffer[150]; //buffer needed for io
            memset(buffer, '\0', sizeof(buffer)); 
            // sprintf(buffer, "Day %i Rank %i: %i places not on fire, %i places on fire, %i burnt locations\n", b, myrank, dayData[0], dayData[1], dayData[2]);
            int count = strlen(buffer); // exclude null terminator
            // Determine the size of the file
            MPI_Offset file_size;
            MPI_File_get_size(file, &file_size);

            // Set the offset to the end of the file
            MPI_Offset offset = file_size + (count * myrank);
            MPI_File_write_at_all(file, offset, buffer, count, MPI_CHAR, MPI_STATUS_IGNORE);
            MPI_File_sync(file);

            // Print total day summary
            if (myrank == 0) {
                char buffer0[150]; //buffer needed for io
                memset(buffer0, '\0', sizeof(buffer)); 
                sprintf(buffer0, "Day %i Totals: %i places not on fire, %i places on fire, %i burnt locations\n", b, dayTotals[0], dayTotals[1], dayTotals[2]);
                int count = strlen(buffer0); // exclude null terminator
                // Determine the size of the file
                MPI_Offset file_size;
                MPI_File_get_size(file, &file_size);

                // Set the offset to the end of the file
                MPI_Offset offset = file_size;
                MPI_File_write_at(file, offset, buffer0, count, MPI_CHAR, MPI_STATUS_IGNORE);
            }
        }

        // loop through infected nodes, find new infections and deaths
        struct Node* n = infectedList.head;
        int messageCount[numranks]; // index = rank, count[index] = number of messages to that rank
        memset( messageCount, 0, numranks*sizeof(int) );
        while (n != NULL) { 
            int id = n->id;
            // check for connections being infected
            for (int c = 0; c < connections; c++) {
                float chance = (double)rand()/RAND_MAX;
                //CHANGE!!!!
                //add the seasons feature to this and after a day has changed make the area go from burning to burnt so dead
                if (chance < spread_chance) {
                    int infectedID = land[id].neighbors[c];
                    // if infected id is owned by this rank, add to self queue
                    if (infectedID % numranks == myrank) {
                        if (land[infectedID].status == 0) {
                            addToList(&infectedQueue, infectedID);
                            land[infectedID].status = 1;
                        }
                    } else { // else, increment corresponding message count and send message
                        messageCount[infectedID % numranks] += 1;
                        MPI_Request request;
                        MPI_Isend(&infectedID, 1, MPI_LONG_LONG_INT, infectedID % numranks, 0, MPI_COMM_WORLD, &request);
                    }
                }
            }

            // after checking for infections by the node, check to see if the node dies
            bool dead = false;
            float chance = (double)rand()/RAND_MAX;
            //CHANGE!!
            //make this different after a day make it stop burning and make it go burnt
            if (chance < extinguish_chance) {
                land[id].status = 2;          
                struct Node* temp = n->next;
                deleteNode(&infectedList, n);
                burntCount += 1;
                n = temp;
                dead = true;
                //printf("Rank %i: Node %i has died\n", myrank, id); // Debug printing
            }

            // increment node along linked list
            // don't increment if node died (already incremented with temp variable)
            if (!dead) {
                n = n->next;
            }
        }
        
        // Reduce messageCount across all ranks to get total message count for this rank (totalMessageCount[myrank])
        int totalMessageCount[numranks];
        MPI_Allreduce(&messageCount, &totalMessageCount, numranks, MPI_LONG_LONG_INT, MPI_SUM,  MPI_COMM_WORLD);

        // Receive all messages and add nodes not infected already to the infectedQueue
        for (int i = 0; i < totalMessageCount[myrank]; i++) {
            int recv;
            MPI_Recv(&recv, 1, MPI_LONG_LONG_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            if (land[recv].status == 0) { // if node is already in self infectedQueue then status = 1, so there will be no repeats
                land[recv].status = 1;
                addToList(&infectedQueue, recv);
            }
        }
        // Loop through queue and add nodes to infectedList
        n = infectedQueue.head;
        while(n != NULL) {
            int id = n->id;
            //printf("Rank %i: Node %i has been infected\n", myrank, id); // Debug printing
            addToList(&infectedList, id);
            deleteNode(&infectedQueue, n);
            n = infectedQueue.head;
        }
    }
    MPI_File_close(&file);
}

int main(int argc, char *argv[]) {

    // mpi init
    MPI_Init(&argc, &argv);
    int myrank;
    int numranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);            
    double start_cycles= clock_now();
    sim(myrank, numranks);
    double end_cycles= clock_now();
    double time_in_secs = ((double)(end_cycles - start_cycles)) / clock_frequency;
    if (myrank == 0) {
        printf("CPU Reduce time: %f\n", time_in_secs);
    }
    MPI_Finalize();
    return 0;
}
