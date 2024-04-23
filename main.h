#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>

#define num_trees 256
#define connections 5
#define spread_chance 0.3
#define extinguish_chance 0.1
#define clock_frequency 512000000
#define startBurningChance 0.05

#define SPRING 0
#define SUMMER 1
#define AUTUMN 2
#define WINTER 3

#define RAINY 0
#define SUNNY 1
#define SNOWY 2
#define CLOUDY 3

#define springBurningMod .7
#define summerBurningMod 1.2
#define autumnBurningMod .5
#define winterBurningMod .3

#define rainyMod .2
#define sunnyMod .8
#define snowyMod .2
#define cloudyMod .5

#define RAINY_SPREAD_MOD 0.5
#define SUNNY_SPREAD_MOD 1.2
#define SNOWY_SPREAD_MOD 0.3
#define CLOUDY_SPREAD_MOD 0.9

#define RAINY_EXTINGUISH_MOD 1.5
#define SUNNY_EXTINGUISH_MOD 0.8
#define SNOWY_EXTINGUISH_MOD 1.2
#define CLOUDY_EXTINGUISH_MOD 1.0

typedef struct Cell {
    int id;
    int status; // 1 = tree, 0 = empty, 2 = burning
    int neighbors[connections];
} Cell;

typedef struct List {
    struct Node* head;
    struct Node* end;
    int size;
} List;

typedef struct Node {
    int id;
    struct Node* next;
    struct Node* prev;
} Node;
