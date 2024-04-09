up:
	gcc -O3 -msse4 -mavx conv-harness.c -fopenmp
run:
	./a.out 64 64 1 64 64