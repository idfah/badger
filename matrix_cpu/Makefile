include ../Makecommon

all: matrix.o

matrix.o: matrix.c matrix.h
	$(CC) $(CFLAGS) $(CPPFLAGS) -I.. $< -c -o $@

clean:
	rm -f matrix.o

remove: clean
