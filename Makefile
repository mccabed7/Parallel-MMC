up:
	gcc -O3 -msse4 conv-harness.c -fopenmp
run:
	./a.out 16 16 1 32 32