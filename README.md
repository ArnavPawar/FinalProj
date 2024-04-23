# Parallel-Programming-Project-S2023
Project repository for RPI' Parallel Programming class in Spring 2023

For the project we will create a parallel discrete event model that will simulate the world going through a pandemic, similar to covid-19.
To create this model, simple structs to represent a person will be created. Each will have an integer representing whether they are alive (1), infected (2), or dead (3) and a list of randomized people representing their social circle that they have the potential to infect if they themselves are infected.

Each struct/person in the model will be assigned to an MPI thread based on the number of threads available at runtime. Each MPI thread will store a list of its active infections, and will run through each infected person it is assigned to.

The model will be based on a cyclical day model, where each day a person has a defined chance to infect each person in their social circle. If a person infected is managed by another thread, a message will be sent to that person’s thread including who has been infected. Otherwise it is stored in a self buffer. Then a message is printed to I/O saying who has been infected and by whom. After infection, each person has a defined chance to die from the infection. The definitions of these chances should be able to be changed easily, using inputs or defined defaults if no inputs are provided. If a person dies, they are removed from the thread’s list of infected people and a message is printed to I/O that they have died.

Once an MPI thread has run through all of its assigned infected people, the day is over. Threads should read self-buffers and MPI buffers and add the corresponding people to their list of infected persons to be used for the following day. This simulation will run until either everyone is infected/dead, or all of the infected people have died off and no more infected people remain.

Definitions used/inputs allowed should be:
-  Chance to infect someone else (happens once for each person in social circle
-  Chance to die from infection (happens once each day)
-  Avg. number of people in social circle (connectedness of graph)
We’ll try to use seeds to reduce randomness between tests, but I have a feeling that as the number of threads change that will impact the randomness and cause different results.

This should be able to scale well to different tests. Tests changing each input can be run for different numbers of threads. This will create many different graphs showing the results of parallelization. Like the MPI homework, tests will be run with:
-  2 MPI ranks (1 compute node)
-  4 MPI ranks (1 compute node)
-  8 MPI ranks (1 compute node)
-  16 MPI ranks (1 compute node)
-  32 MPI ranks (1 compute node)
-  64 MPI ranks (2 compute nodes)
-  128 MPI ranks (4 compute nodes)
-  256 MPI ranks (8 compute nodes)

# Complile/Run Instructions
mpicc main.c -o a.out
mpirun -N 2 a.out

# Github link
https://github.com/EthanTuttle/Parallel-Programming-Project-S2023