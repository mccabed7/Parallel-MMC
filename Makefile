up:
	gcc -O3 -msse4 -mavx -mavx512f -mavx512dq conv-harness.c -fopenmp
run:
	./a.out 100 100 1 32 128