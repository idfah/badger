include ../Makecommon

all: matrix.o

matrix.o: matrix.cu matrix.h ../benchmark.h ../errcheck.h
	$(NVCC) $(NVCFLAGS) $(CPPFLAGS) -I.. $< -c -o $@

clean:
	rm -f matrix.o

remove: clean
