CXX ?= g++
CC ?= gcc
CFLAGS = -g -Wall -Wconversion -fPIC -lpthread
MEX = /opt/local-ubuntu/MATLAB/R2011a/bin/mex
MEXFLAGS = -g CC\#$(CXX) CXX\#$(CXX) CFLAGS\#"$(CFLAGS)" CXXFLAGS\#"$(CFLAGS)" -largeArrayDims
LIBS = blas/blas.a

all: x_do_train_one_vs_all.mexa64 x_do_train_one_vs_one.mexa64 x_do_classify.mexa64

x_do_train_one_vs_all.mexa64: x_do_train_one_vs_all.c x_common.c x_common.h x_defines.h tron.o linear.o blas/blas.a
	$(MEX) $(MEXFLAGS) x_do_train_one_vs_all.c x_common.c tron.o linear.o $(LIBS)

x_do_train_one_vs_one.mexa64: x_do_train_one_vs_one.c x_common.c x_common.h x_defines.h tron.o linear.o blas/blas.a
	$(MEX) $(MEXFLAGS) x_do_train_one_vs_one.c x_common.c tron.o linear.o $(LIBS)

x_do_classify.mexa64: x_do_classify.c x_common.c x_common.h x_defines.h tron.o linear.o blas/blas.a
	$(MEX) $(MEXFLAGS) x_do_classify.c x_common.c tron.o linear.o $(LIBS)

tron.o: tron.cpp tron.h
	$(CXX) $(CFLAGS) -c -o tron.o tron.cpp

linear.o: linear.cpp linear.h
	$(CXX) $(CFLAGS) -c -o linear.o linear.cpp

blas/blas.a: blas/*.c blas/*.h
	make -C blas OPTFLAGS='$(CFLAGS)' CC='$(CC)'

clean:
	make -C blas clean
	rm -f *.mexa64
	rm -f *.o
