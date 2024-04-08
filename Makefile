up:
	gcc -O3 -msse4 -mavx -mavx512dq conv-harness.c -fopenmp
run:
	./a.out 64 64 5 128 128