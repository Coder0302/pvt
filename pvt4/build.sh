#!/bin/bash

gcc -o par paralelN-body.c -fopenmp -lm
gcc -o atom atomicN-body.c -fopenmp -lm
gcc -o blok n-blockN-body.c -fopenmp -lm
gcc -o cal calculatN-body.c -fopenmp -lm
gcc -o mem memoryN-body.c -fopenmp -lm

./par 10 gnuplot/prog-paralel.dat
./atom 10 gnuplot/prog-atomic.dat
./blok 10 gnuplot/prog-nblock.dat
./cal 10 gnuplot/prog-calculat.dat
./mem 10 gnuplot/prog-memory.dat


gnuplot gnuplot/graph_paralel.gp