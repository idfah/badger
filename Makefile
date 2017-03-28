######################
# Badger build rules #
#                    #
# by                 #
#   Elliott Forney   #
#   5.3.2010         #
######################

#
# setup variables
#

# include common settings
include Makecommon

APPS :=
APPS_CPU := mult_cpu add_cpu sqr_cpu scl_cpu pmult_cpu trans_cpu xor_cpu randnn_cpu phiprime_cpu sinewave_cpu
APPS_GPU := mult_gpu add_gpu sqr_gpu scl_gpu pmult_gpu trans_gpu xor_gpu randnn_gpu phiprime_gpu sinewave_gpu

MODS := benchmark.o errcheck.o
MODS_CPU := matrix_cpu.o ffnn_cpu.o
MODS_GPU := matrix_gpu.o ffnn_gpu.o

CPPFLAGS_CPU := -I./matrix_cpu
CPPFLAGS_GPU := -I./matrix_gpu

#
# check environment for options
#

# check for cuda device emulation
ifdef CUDA_EMU
	NVCFLAGS += -deviceemu -DEMU
endif

# check for debug level
ifdef DEBUG
	NVCFLAGS += -DDEBUG=99
	CFLAGS   += -DDEBUG=99
endif

#
# make everything
#

all: $(APPS) $(APPS_CPU) $(APPS_GPU)

#
# common modules
#

benchmark.o: benchmark.c benchmark.h
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -c -o $@

errcheck.o: errcheck.cu errcheck.h
	$(NVCC) $(NVCFLAGS) $(CPPFLAGS) $< -c -o $@

#
# gpu modules
#

matrix_gpu.o: 
	@ cd matrix_gpu; $(MAKE) $(MFLAGS)
	@ cp matrix_gpu/matrix.o matrix_gpu.o

ffnn_gpu.o: ffnn.c ffnn.h errcheck.o matrix_gpu.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CPPFLAGS_GPU) $< -c -o $@

#
# cpu modules
#

matrix_cpu.o:
	@ cd matrix_cpu; $(MAKE) $(MFLAGS)
	@ cp matrix_cpu/matrix.o matrix_cpu.o

ffnn_cpu.o: ffnn.c ffnn.h errcheck.o matrix_gpu.o
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CPPFLAGS_CPU) $< -c -o $@

#
# gpu apps
#

mult_gpu: mult.c $(MODS) $(MODS_GPU)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CPPFLAGS_GPU) $< -o $@ $(MODS) $(MODS_GPU) $(LDFLAGS) $(MKL_CPPFLAGS) $(MKL_LDFLAGS)

add_gpu: add.c $(MODS) $(MODS_GPU)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CPPFLAGS_GPU) $< -o $@ $(MODS) $(MODS_GPU) $(LDFLAGS)

sqr_gpu: sqr.c $(MODS) $(MODS_GPU)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CPPFLAGS_GPU) $< -o $@ $(MODS) $(MODS_GPU) $(LDFLAGS)

scl_gpu: scl.c $(MODS) $(MODS_GPU)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CPPFLAGS_GPU) $< -o $@ $(MODS) $(MODS_GPU) $(LDFLAGS)

pmult_gpu: pmult.c $(MODS) $(MODS_GPU)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CPPFLAGS_GPU) $< -o $@ $(MODS) $(MODS_GPU) $(LDFLAGS)

trans_gpu: trans.c $(MODS) $(MODS_GPU)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CPPFLAGS_GPU) $< -o $@ $(MODS) $(MODS_GPU) $(LDFLAGS)

xor_gpu: xor.c $(MODS) $(MODS_GPU)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CPPFLAGS_GPU) $< -o $@ $(MODS) $(MODS_GPU) $(LDFLAGS)

randnn_gpu: randnn.c $(MODS) $(MODS_GPU)
	$(CC) -DGPU $(CFLAGS) $(CPPFLAGS) $(CPPFLAGS_GPU) $< -o $@ $(MODS) $(MODS_GPU) $(LDFLAGS)

phiprime_gpu: phiprime.c $(MODS) $(MODS_GPU)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CPPFLAGS_GPU) $< -o $@ $(MODS) $(MODS_GPU) $(LDFLAGS)

sinewave_gpu: sinewave.c $(MODS) $(MODS_GPU)
	$(CC) -DGPU $(CFLAGS) $(CPPFLAGS) $(CPPFLAGS_GPU) $< -o $@ $(MODS) $(MODS_GPU) $(LDFLAGS)

#
# cpu apps
#

mult_cpu: mult.c $(MODS) $(MODS_CPU)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CPPFLAGS_CPU) $< -o $@ $(MODS) $(MODS_CPU) $(LDFLAGS) $(MKL_CPPFLAGS) $(MKL_LDFLAGS)

add_cpu: add.c $(MODS) $(MODS_CPU)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CPPFLAGS_CPU) $< -o $@ $(MODS) $(MODS_CPU) $(LDFLAGS) $(MKL_CPPFLAGS) $(MKL_LDFLAGS)

sqr_cpu: sqr.c $(MODS) $(MODS_CPU)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CPPFLAGS_CPU) $< -o $@ $(MODS) $(MODS_CPU) $(LDFLAGS) $(MKL_CPPFLAGS) $(MKL_LDFLAGS)

scl_cpu: scl.c $(MODS) $(MODS_CPU)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CPPFLAGS_CPU) $< -o $@ $(MODS) $(MODS_CPU) $(LDFLAGS) $(MKL_CPPFLAGS) $(MKL_LDFLAGS)

pmult_cpu: pmult.c $(MODS) $(MODS_CPU)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CPPFLAGS_CPU) $< -o $@ $(MODS) $(MODS_CPU) $(LDFLAGS) $(MKL_CPPFLAGS) $(MKL_LDFLAGS)

trans_cpu: trans.c $(MODS) $(MODS_CPU)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CPPFLAGS_CPU) $< -o $@ $(MODS) $(MODS_CPU) $(LDFLAGS) $(MKL_CPPFLAGS) $(MKL_LDFLAGS)

xor_cpu: xor.c $(MODS) $(MODS_CPU)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CPPFLAGS_CPU) $< -o $@ $(MODS) $(MODS_CPU) $(LDFLAGS) $(MKL_CPPFLAGS) $(MKL_LDFLAGS)

randnn_cpu: randnn.c $(MODS) $(MODS_CPU)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CPPFLAGS_CPU) $< -o $@ $(MODS) $(MODS_CPU) $(LDFLAGS) $(MKL_CPPFLAGS) $(MKL_LDFLAGS)

phiprime_cpu: phiprime.c $(MODS) $(MODS_CPU)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CPPFLAGS_CPU) $< -o $@ $(MODS) $(MODS_CPU) $(LDFLAGS) $(MKL_CPPFLAGS) $(MKL_LDFLAGS)

sinewave_cpu: sinewave.c $(MODS) $(MODS_CPU)
	$(CC) $(CFLAGS) $(CPPFLAGS) $(CPPFLAGS_CPU) $< -o $@ $(MODS) $(MODS_CPU) $(LDFLAGS) $(MKL_CPPFLAGS) $(MKL_LDFLAGS)

#
# clean things up
#

# remove modules
clean:
	rm -f $(MODS) $(MODS_CPU) $(MODS_GPU)
	@ cd matrix_cpu; $(MAKE) $(MKFLAGS) clean
	@ cd matrix_gpu; $(MAKE) $(MKFLAGS) clean

# remove modules and apps
remove: clean
	rm -f $(APPS) $(APPS_CPU) $(APPS_GPU)
	@ cd matrix_cpu; $(MAKE) $(MKFLAGS) remove
	@ cd matrix_gpu; $(MAKE) $(MKFLAGS) remove

.PHONY: all clean remove matrix_cpu.o matrix_gpu.o
