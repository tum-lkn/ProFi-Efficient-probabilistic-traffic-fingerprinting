phmm : array_func.o forward.o backward.o update.o learning.o
	gcc -shared -o phmm.so -fPIC array_func.o forward.o backward.o update.o learning.o

array_func.o :  array_func.c
	gcc -c -fPIC array_func.c

forward.o : forward.c
	gcc -c -fPIC forward.c

backward.o : backward.c
	gcc -c -fPIC backward.c

update.o : update.c
	gcc -c -fPIC update.c

learning.o : learning.c
	gcc -c -fPIC learning.c

mc : mc.o
	gcc -shared -o mc.so -fPIC mc.o

mc.o : mc.c
	gcc -c -fPIC mc.c

clean :
	rm array_func.o backward.o forward.o learning.o update.o phmm.so

