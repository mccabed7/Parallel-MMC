up:
	gcc -O3 -msse4 -mavx conv-harness.c -fopenmp
run:
	./a.out 128 128 7 256 256