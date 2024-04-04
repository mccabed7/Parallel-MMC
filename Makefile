up:
	gcc -O3 -msse4 conv-harness.c -fopenmp
run:
	./a.out 100 100 5 1 28 128