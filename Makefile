CC = riscv64-unknown-linux-gnu-gcc
CFLAGS := -O3 -static -DRISCV

#CC = gcc
#CFLAGS := -O3 -std=gnu99

qsort: afterBootTest.c util.h
	${CC} ${CFLAGS} -o afterBootTest afterBootTest.c -fopenmp

clean:
	rm -f afterBootTest
