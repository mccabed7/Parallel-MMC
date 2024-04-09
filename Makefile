up:
	gcc -O3 -msse4 -mavx conv-harness.c -fopenmp
run:
	./a.out 40 64 3 32 128