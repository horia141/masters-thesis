CXX ?= g++
CC ?= gcc
CFLAGS = -g -Wall -Wconversion -fPIC -lpthread
MEX = /opt/local-ubuntu/MATLAB/R2011a/bin/mex
MEXFLAGS = -g CC\#$(CXX) CXX\#$(CXX) CFLAGS\#"$(CFLAGS)" CXXFLAGS\#"$(CFLAGS)" -largeArrayDims
LIBS = blas/blas.a

all: x_do_train_one_vs_all x_do_train_one_vs_one x_do_classify

x_do_train_one_vs_all: x_do_train_one_vs_all.c x_common.h tron.o linear.o blas/blas.a
	$(MEX) $(MEXFLAGS) x_do_train_one_vs_all.c tron.o linear.o $(LIBS)

x_do_train_one_vs_one: x_do_train_one_vs_one.c x_common.h tron.o linear.o blas/blas.a
	$(MEX) $(MEXFLAGS) x_do_train_one_vs_one.c tron.o linear.o $(LIBS)

x_do_classify: x_do_classify.c x_common.h tron.o linear.o blas/blas.a
	$(MEX) $(MEXFLAGS) x_do_classify.c tron.o linear.o $(LIBS)

tron.o: tron.cpp tron.h
	$(CXX) $(CFLAGS) -c -o tron.o tron.cpp

linear.o: linear.cpp linear.h
	$(CXX) $(CFLAGS) -c -o linear.o linear.cpp

blas/blas.a: blas/*.c blas/*.h
	make -C blas OPTFLAGS='$(CFLAGS)' CC='$(CC)'

clean:
	make -C blas clean
	rm -f *~ tron.o linear.o
