up:
	gcc -O3 -msse4 -mavx conv-harness.c -fopenmp
run:
	./a.out 100 100 3 64 128