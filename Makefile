up:
	gcc -O3 -msse4 conv-harness.c -fopenmp
run:
	./a.out 100 100 1 64 128