#include "main.h"
#include <mpi.h>
#include "clockcycle.h"

extern void addToList(struct Node* n, struct List *li, int val, int rank);
extern void deleteNode(struct List *li, struct Node* n, int rank);
extern const char* getSeasonName(int val);

void initializeList(List *li) {
    li->head = NULL;
    li->end = NULL;
    li->size = 0;
}

double calculateInfectionChance(int season, int dayType, double baseChance) {
    switch(season) {
        case SPRING: baseChance *= springBurningMod; break;
        case SUMMER: baseChance *= summerBurningMod; break;
        case AUTUMN: baseChance *= autumnBurningMod; break;
        case WINTER: baseChance *= winterBurningMod; break;
    }

    switch(dayType) {
        case RAINY: baseChance *= rainyMod; break;
        case SUNNY: baseChance *= sunnyMod; break;
        case SNOWY: baseChance *= snowyMod; break;
        case CLOUDY: baseChance *= cloudyMod; break;
    }

    return baseChance;
}

void initializeNeighbors(int *neighbors, int numTrees) {
    for (int c = 0; c < connections; c++) {
        neighbors[c] = rand() % numTrees;
    }
}

void writeFinalSummary(MPI_File file, int day, int *dayTotals) {
    char buffer[150];  // Buffer for I/O operations
    // Clear the buffer to avoid any garbage values
    memset(buffer, 0, sizeof(buffer));

    // Prepare the summary message
    sprintf(buffer, "Day %i Totals: %i places not on fire, %i places on fire, %i burnt locations\nEnd of Sim\n", 
            day, dayTotals[0], dayTotals[1], dayTotals[2]);
    int count = strlen(buffer); // Get the size of the buffer to be written

    // Determine the size of the file to calculate the offset for appending
    MPI_Offset offset;
    MPI_File_get_size(file, &offset);  // Get the current file size

    // Write the summary to the file at the calculated offset
    MPI_File_write_at(file, offset, buffer, count, MPI_CHAR, MPI_STATUS_IGNORE);
}

void writeDaySummary(MPI_File file, int day, int *dayTotals, const char *weatherDescription) {
    char buffer[150];
    // Prepare the message
    sprintf(buffer, "Day %i Totals: %i places not on fire, %i places on fire, %i burnt locations, %s type of day\n",
            day, dayTotals[0], dayTotals[1], dayTotals[2], weatherDescription);
    int count = strlen(buffer); // Get the size of the buffer to be written

    // Determine the size of the file to calculate the offset for appending
    MPI_Offset offset;
    MPI_File_get_size(file, &offset); // Get the current file size

    // Write the summary to the file at the calculated offset
    MPI_File_write_at(file, offset, buffer, count, MPI_CHAR, MPI_STATUS_IGNORE);
}

int main(int argc, char *argv[]) {

    // mpi init
    MPI_Init(&argc, &argv);
    int myrank, numranks;
    int burntCount = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);            
    double start_cycles= clock_now();
    
    // initialize rank vars
    Cell *land = calloc(num_trees, sizeof(struct Cell));
    if (land == NULL) {
        fprintf(stderr, "Failed to allocate memory for land.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    List infectedList, infectedQueue;
    initializeList(&infectedList);
    initializeList(&infectedQueue);

    MPI_File file;
    if (MPI_File_open(MPI_COMM_WORLD, "output.txt", MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file) != MPI_SUCCESS) {
        fprintf(stderr, "[%d] Failed to open file for output.\n", myrank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }    
    srand(myrank);
    int id = myrank; // the id of the Cell we're creating

    int season = rand() % 4;  // Random season: 0 to 3
    const char* seasonSTR = getSeasonName(season);
    char introBuffer[100];  // Adjust size as necessary
    sprintf(introBuffer, "Season Introduction: The current season is %s.\n", seasonSTR);

    //CAHNGE!!!!
    //make a random function call that will decide what season it is outside
    while (id < num_trees) {
        // generate list of connections
        int neighbors[connections];
        initializeNeighbors(currentCell->neighbors, num_trees);
        // for (int c = 0; c < connections; c++) {
        //     neighbors[c] = rand() % num_trees; // rand() % x = random number from 0 to x-1 ( e.g. [0-X) )
        // }
        // determine if node starts infected
        double chance = (double)rand()/RAND_MAX;

        int dayType = rand() % 4; // Random day type: 0 to 3

        int isInfected = 0;

        double modifiedChance = calculateInfectionChance(season, dayType, startBurningChance);
        // double modifiedChance = startBurningChance;

        //CHANGE!!!!
        // make this change the chance of starting burning based on the weather outside
        if (chance < modifiedChance) {
            struct Node* t = malloc(sizeof(struct Node));
            addToList(t, &infectedList, id, myrank);
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
    bool simContinue=true;
    int day = 0;
    while (simContinue) {
        // Beginning of day information
        day += 1;
        int healthyCount = (num_trees / numranks) - infectedList.size - burntCount;
        int dayData[3] = {healthyCount, infectedList.size, burntCount};
        int dayTotals[3];
        MPI_Allreduce(&dayData, &dayTotals, 3, MPI_LONG_LONG_INT, MPI_SUM, MPI_COMM_WORLD);
        // Day summaries

        if (dayTotals[0] == 0 || dayTotals[1] == 0) { // sim ends when either no infected or non-infected cell remain

            // Print final individual rank day summary
            char buffer[150]; //buffer needed for io
            memset(buffer, '\0', sizeof(buffer)); 
            // sprintf(buffer, "End of Sim Day %i Rank %i: %i places not on fire, %i places on fire, %i burnt locations\n", day, myrank, dayData[0], dayData[1], dayData[2]);
            int count = strlen(buffer); // exclude null terminator
            // Determine the size of the file
            MPI_Offset file_size;
            MPI_File_get_size(file, &file_size);

            // Set the offset to the end of the file
            offset += count * myrank; // Calculate offset for each rank to avoid overlap
            MPI_File_write_at_all(file, offset, buffer, count, MPI_CHAR, MPI_STATUS_IGNORE);
            MPI_File_sync(file);

            // Print final total day summary
            if (myrank == 0) {
                // char buffer0[150]; //buffer needed for io
                // memset(buffer0, '\0', sizeof(buffer)); 
                // sprintf(buffer0, "Day %i Totals: %i places not on fire, %i places on fire, %i burnt locations\nEnd of Sim\n", day, dayTotals[0], dayTotals[1], dayTotals[2]);
                // count = strlen(buffer0); // exclude null terminator
                // // Determine the size of the file
                // MPI_Offset file_size;
                // MPI_File_get_size(file, &file_size);
                // // Set the offset to the end of the file
                // MPI_Offset offset = file_size;
                // MPI_File_write_at(file, offset, buffer0, count, MPI_CHAR, MPI_STATUS_IGNORE);
                writeFinalSummary(file, day, dayTotals);
            }
            break;
        } 
        else {
            int weather = rand() % 4;  // Determine today's weather randomly
            char weatherDescription[20];  // Buffer to store description of the weather
            switch (weather) {
                case RAINY:
                    strcpy(weatherDescription, "Rainy");
                    break;
                case SUNNY:
                    strcpy(weatherDescription, "Sunny");
                    break;
                case SNOWY:
                    strcpy(weatherDescription, "Snowy");
                    break;
                case CLOUDY:
                    strcpy(weatherDescription, "Cloudy");
                    break;
            }
            // Print individual rank day summary
            char buffer[150]; //buffer needed for io
            memset(buffer, '\0', sizeof(buffer)); 
            // sprintf(buffer, "Day %i Rank %i: %i places not on fire, %i places on fire, %i burnt locations\n", day, myrank, dayData[0], dayData[1], dayData[2]);
            int count = strlen(buffer); // exclude null terminator
            // Determine the size of the file
            // MPI_Offset file_size;
            // MPI_File_get_size(file, &file_size);

            // // Set the offset to the end of the file
            // MPI_Offset offset = file_size + (count * myrank);
            // MPI_File_write_at_all(file, offset, buffer, count, MPI_CHAR, MPI_STATUS_IGNORE);
            // MPI_File_sync(file);
            MPI_Offset offset;
            MPI_File_get_size(file, &offset);  // Get current file size for appending
            offset += count * myrank; // Calculate offset to prevent overlap among ranks
            MPI_File_write_at_all(file, offset, buffer, count, MPI_CHAR, MPI_STATUS_IGNORE);
            MPI_File_sync(file); 

            // Print total day summary
            if (myrank == 0) {
                writeDaySummary(file, day, dayTotals, weatherDescription);
                // char buffer0[150]; //buffer needed for io
                // memset(buffer0, '\0', sizeof(buffer)); 
                // sprintf(buffer0, "Day %i Totals: %i places not on fire, %i places on fire, %i burnt locations, %s type of day\n", day, dayTotals[0], dayTotals[1], dayTotals[2],weatherDescription);
                // int count = strlen(buffer0); // exclude null terminator
                // // Determine the size of the file
                // MPI_Offset file_size;
                // MPI_File_get_size(file, &file_size);

                // // Set the offset to the end of the file
                // MPI_Offset offset = file_size;
                // MPI_File_write_at(file, offset, buffer0, count, MPI_CHAR, MPI_STATUS_IGNORE);
            }
        }
//START FROM HERE
        // loop through infected nodes, find new infections and deaths
        struct Node* n = infectedList.head;
        int messageCount[numranks]; // index = rank, count[index] = number of messages to that rank
        memset( messageCount, 0, numranks*sizeof(int) );
        while (n != NULL) { 
            int id = n->id;

            int weather = rand() % 4;  // Determine today's weather randomly
            double spread_mod = 1.0, extinguish_mod = 1.0;

            switch (weather) {
                case RAINY:
                    spread_mod = RAINY_SPREAD_MOD;
                    extinguish_mod = RAINY_EXTINGUISH_MOD;
                    break;
                case SUNNY:
                    spread_mod = SUNNY_SPREAD_MOD;
                    extinguish_mod = SUNNY_EXTINGUISH_MOD;
                    break;
                case SNOWY:
                    spread_mod = SNOWY_SPREAD_MOD;
                    extinguish_mod = SNOWY_EXTINGUISH_MOD;
                    break;
                case CLOUDY:
                    spread_mod = CLOUDY_SPREAD_MOD;
                    extinguish_mod = CLOUDY_EXTINGUISH_MOD;
                    break;
            }

            double daily_spread_chance = spread_chance * spread_mod;
            double daily_extinguish_chance = extinguish_chance * extinguish_mod;

            // check for connections being infected
            for (int c = 0; c < connections; c++) {
                float chance = (double)rand()/RAND_MAX;
                //CHANGE!!!!
                //add the seasons feature to this and after a day has changed make the area go from burning to burnt so dead
                if (chance < daily_spread_chance) {
                    int infectedID = land[id].neighbors[c];
                    // if infected id is owned by this rank, add to self queue
                    if (infectedID % numranks == myrank) {
                        if (land[infectedID].status == 0) {
                            struct Node* t = malloc(sizeof(struct Node));
                            addToList(t, &infectedQueue, infectedID, myrank);
                            land[infectedID].status = 1;
                            
                        }
                    } 
                    else { // else, increment corresponding message count and send message
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
            if (chance < daily_extinguish_chance) {
                land[id].status = 2;          
                struct Node* temp = n->next;
                deleteNode(&infectedList, n, myrank);
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
                struct Node* t = malloc(sizeof(struct Node));
                addToList(t, &infectedQueue, recv, myrank);
                
            }
        }
        // Loop through queue and add nodes to infectedList
        n = infectedQueue.head;
        while(n != NULL) {
            int id = n->id;
            //printf("Rank %i: Node %i has been infected\n", myrank, id); // Debug printing
            struct Node* t = malloc(sizeof(struct Node));
            addToList(t, &infectedList, id, myrank);
            
            deleteNode(&infectedQueue, n, myrank);
            n = infectedQueue.head;
        }
    }
    MPI_File_close(&file);
    
    double end_cycles= clock_now();
    double time_in_secs = ((double)(end_cycles - start_cycles)) / clock_frequency;
    if (myrank == 0) {
        printf("CPU Reduce time: %f\n", time_in_secs);
    }
    MPI_Finalize();
    return 0;
}