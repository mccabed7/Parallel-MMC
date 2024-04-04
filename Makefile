up:
	gcc -O3 -msse4 conv-harness.c -fopenmp
run:
	./a.out 100 100 3 32 128