CXX ?= g++
CC ?= gcc
CFLAGS = -g -Wall -Wconversion -fPIC -lpthread
MEX = /opt/local-ubuntu/MATLAB/R2011a/bin/mex
MEXFLAGS = -g CC\#$(CXX) CXX\#$(CXX) CFLAGS\#"$(CFLAGS)" CXXFLAGS\#"$(CFLAGS)" -largeArrayDims

all: x_do_train_one_vs_all.mexa64 x_do_train_one_vs_one.mexa64 x_do_classify.mexa64

x_do_train_one_vs_all.mexa64: x_do_train_one_vs_all.c x_common.h svm.o
	$(MEX) $(MEXFLAGS) x_do_train_one_vs_all.c svm.o

x_do_train_one_vs_one.mexa64: x_do_train_one_vs_one.c x_common.h svm.o
	$(MEX) $(MEXFLAGS) x_do_train_one_vs_one.c svm.o

x_do_classify.mexa64: x_do_classify.c x_common.h svm.o
	$(MEX) $(MEXFLAGS) x_do_classify.c svm.o

svm.o: svm.cpp svm.h
	$(CXX) $(CFLAGS) -c -o svm.o svm.cpp

clean:
	rm -f *.mexa64
	rm -f *.o
