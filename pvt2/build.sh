#!/bin/bash

gcc -o main main.c -std=c99 -Wall -O2 -fopenmp

./main

gnuplot gnuplot/graph.gp