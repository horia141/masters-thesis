CXX ?= g++
CC ?= gcc
CFLAGS = -g -Wall -Wconversion -fPIC -lpthread -D_GNU_SOURCE
MEX = /opt/local-ubuntu/MATLAB/R2011a/bin/mex
MEXFLAGS = -g CC\#$(CXX) CXX\#$(CXX) CFLAGS\#"$(CFLAGS)" CXXFLAGS\#"$(CFLAGS)" -largeArrayDims
XTERN = +xtern

all: +classifiers/+liblinear/x_do_train_one_vs_all.mexa64 +classifiers/+liblinear/x_do_train_one_vs_one.mexa64 +classifiers/+liblinear/x_do_classify.mexa64 +classifiers/+libsvm/x_do_train_one_vs_all.mexa64 +classifiers/+libsvm/x_do_train_one_vs_one.mexa64 +classifiers/+libsvm/x_do_classify.mexa64 +transforms/+record/dictionary_correlation_order.mexa64

+classifiers/+liblinear/x_do_train_one_vs_all.mexa64: +classifiers/+liblinear/x_do_train_one_vs_all.c +classifiers/+liblinear/x_defines.h +classifiers/+liblinear/tron.o +classifiers/+liblinear/linear.o +classifiers/+liblinear/blas/blas.a +xtern/x_common.c +xtern/x_common.h 
	$(MEX) $(MEXFLAGS) -outdir +classifiers/+liblinear -I+xtern +classifiers/+liblinear/x_do_train_one_vs_all.c +classifiers/+liblinear/blas/blas.a +classifiers/+liblinear/tron.o +classifiers/+liblinear/linear.o +xtern/x_common.c $(LIBS)

+classifiers/+liblinear/x_do_train_one_vs_one.mexa64: +classifiers/+liblinear/x_do_train_one_vs_one.c +classifiers/+liblinear/x_defines.h +classifiers/+liblinear/tron.o +classifiers/+liblinear/linear.o +classifiers/+liblinear/blas/blas.a +xtern/x_common.c +xtern/x_common.h
	$(MEX) $(MEXFLAGS) -outdir +classifiers/+liblinear -I+xtern +classifiers/+liblinear/x_do_train_one_vs_one.c +classifiers/+liblinear/blas/blas.a +classifiers/+liblinear/tron.o +classifiers/+liblinear/linear.o +xtern/x_common.c $(LIBS)

+classifiers/+liblinear/x_do_classify.mexa64: +classifiers/+liblinear/x_do_classify.c +classifiers/+liblinear/x_defines.h +classifiers/+liblinear/tron.o +classifiers/+liblinear/linear.o +classifiers/+liblinear/blas/blas.a +xtern/x_common.c +xtern/x_common.h
	$(MEX) $(MEXFLAGS) -outdir +classifiers/+liblinear -I+xtern +classifiers/+liblinear/x_do_classify.c +classifiers/+liblinear/blas/blas.a +classifiers/+liblinear/tron.o +classifiers/+liblinear/linear.o +xtern/x_common.c $(LIBS)

+classifiers/+liblinear/tron.o: +classifiers/+liblinear/tron.cpp +classifiers/+liblinear/tron.h
	$(CXX) $(CFLAGS) -c -o +classifiers/+liblinear/tron.o +classifiers/+liblinear/tron.cpp

+classifiers/+liblinear/linear.o: +classifiers/+liblinear/linear.cpp +classifiers/+liblinear/linear.h
	$(CXX) $(CFLAGS) -c -o +classifiers/+liblinear/linear.o +classifiers/+liblinear/linear.cpp

+classifiers/+liblinear/blas/blas.a: +classifiers/+liblinear/blas/*.c +classifiers/+liblinear/blas/*.h
	make -C +classifiers/+liblinear/blas OPTFLAGS='$(CFLAGS)' CC='$(CC)'

+classifiers/+libsvm/x_do_train_one_vs_all.mexa64: +classifiers/+libsvm/x_do_train_one_vs_all.c +classifiers/+libsvm/x_defines.h +classifiers/+libsvm/svm.o +xtern/x_common.c +xtern/x_common.h
	$(MEX) $(MEXFLAGS) -outdir +classifiers/+libsvm -I+xtern +classifiers/+libsvm/x_do_train_one_vs_all.c +classifiers/+libsvm/svm.o +xtern/x_common.c

+classifiers/+libsvm/x_do_train_one_vs_one.mexa64: +classifiers/+libsvm/x_do_train_one_vs_one.c +classifiers/+libsvm/x_defines.h +classifiers/+libsvm/svm.o +xtern/x_common.c +xtern/x_common.h
	$(MEX) $(MEXFLAGS) -outdir +classifiers/+libsvm -I+xtern +classifiers/+libsvm/x_do_train_one_vs_one.c +classifiers/+libsvm/svm.o +xtern/x_common.c

+classifiers/+libsvm/x_do_classify.mexa64: +classifiers/+libsvm/x_do_classify.c +classifiers/+libsvm/x_defines.h +classifiers/+libsvm/svm.o +xtern/x_common.c +xtern/x_common.h
	$(MEX) $(MEXFLAGS) -outdir +classifiers/+libsvm -I+xtern +classifiers/+libsvm/x_do_classify.c +classifiers/+libsvm/svm.o +xtern/x_common.c

+classifiers/+libsvm/svm.o: +classifiers/+libsvm/svm.cpp +classifiers/+libsvm/svm.h
	$(CXX) $(CFLAGS) -c -o +classifiers/+libsvm/svm.o +classifiers/+libsvm/svm.cpp

+transforms/+record/dictionary_correlation_order.mexa64: +transforms/+record/dictionary_correlation_order.c +xtern/x_common.c +xtern/x_common.h
	$(MEX) $(MEXFLAGS) -outdir +transforms/+record -I+xtern +transforms/+record/dictionary_correlation_order.c +xtern/x_common.c

clean:
	make -C +classifiers/+liblinear/blas clean
	rm -f +classifiers/+liblinear/*.mexa64
	rm -f +classifiers/+liblinear/*.o
	rm -f +classifiers/+libsvm/*.mexa64
	rm -f +classifiers/+libsvm/*.o
