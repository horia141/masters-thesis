CXX ?= g++
CC ?= gcc
MEX = /opt/MatLAB/R2012a/bin/mex
ACML_LIB_PATH = +xtern/acml/32bit
ACML_INCLUDE_PATH = +xtern/acml/32bit
ACML_LIBS = -lacml
# MEX = /opt/local-ubuntu/MATLAB/R2011a/bin/mex
# ACML_LIB_PATH = +xtern/acml/64bit
# ACML_INCLUDE_PATH = +xtern/acml/64bit
# ACML_LIBS = -lacml -lacml_mv
CFLAGS = -fstrict-aliasing -Wstrict-aliasing -g -Wall -Wconversion -fPIC -L$(ACML_LIB_PATH) -I$(ACML_INCLUDE_PATH) -D_GNU_SOURCE
MEXFLAGS = -g CC\#$(CXX) CXX\#$(CXX) CFLAGS\#"$(CFLAGS)" CXXFLAGS\#"$(CFLAGS)" -largeArrayDims
XTERN_BASE_H = +xtern/base_defines.h +xtern/latools.h +xtern/coding_methods.h +xtern/image_coder.h +xtern/task_control.h
XTERN_BASE_C = +xtern/latools.c +xtern/coding_methods.c +xtern/image_coder.c +xtern/task_control.c
XTERN_H = +xtern/x_mex_interface.h $(XTERN_BASE_H)
XTERN_C = +xtern/x_mex_interface.c $(XTERN_BASE_C)

all: +xtern/test +xtern/liblinear/tron.o +xtern/liblinear/linear.o +xtern/liblinear/blas/blas.a +xtern/x_classifiers_liblinear_classify.mexa64 +xtern/x_classifiers_liblinear_train_one_vs_all.mexa64 +xtern/x_classifiers_liblinear_train_one_vs_one.mexa64 +xtern/x_dictionary_correlation.mexa64 +xtern/x_dictionary_matching_pursuit.mexa64 +xtern/x_dictionary_orthogonal_matching_pursuit.mexa64 +xtern/x_dictionary_optimized_orthogonal_matching_pursuit.mexa64 +xtern/x_image_recoder_code.mexa64

+xtern/test: +xtern/test.c $(XTERN_BASE_C) $(XTERN_BASE_H)
	gcc $(CFLAGS) -o +xtern/test -I+xtern +xtern/test.c $(XTERN_BASE_C) -lpthread -lm $(ACML_LIBS)

+xtern/liblinear/tron.o: +xtern/liblinear/tron.cpp +xtern/liblinear/tron.h
	$(CXX) $(CFLAGS) -c -o +xtern/liblinear/tron.o +xtern/liblinear/tron.cpp

+xtern/liblinear/linear.o: +xtern/liblinear/linear.cpp +xtern/liblinear/linear.h
	$(CXX) $(CFLAGS) -c -o +xtern/liblinear/linear.o +xtern/liblinear/linear.cpp

+xtern/liblinear/blas/blas.a: +xtern/liblinear/blas/*.c +xtern/liblinear/blas/*.h
	make -C +xtern/liblinear/blas OPTFLAGS='$(CFLAGS)' CC='$(CC)'

+xtern/x_classifiers_liblinear_classify.mexa64: +xtern/x_classifiers_liblinear_classify.c +xtern/x_classifiers_liblinear_defines.h +xtern/liblinear/tron.o +xtern/liblinear/linear.o +xtern/liblinear/blas/blas.a $(XTERN_H) $(XTERN_C)
	$(MEX) $(MEXFLAGS) -outdir +xtern -I+xtern +xtern/x_classifiers_liblinear_classify.c +xtern/liblinear/blas/blas.a +xtern/liblinear/tron.o +xtern/liblinear/linear.o $(XTERN_C) -L$(ACML_LIB_PATH) -I$(ACML_INCLUDE_PATH) $(ACML_LIBS)

+xtern/x_classifiers_liblinear_train_one_vs_all.mexa64: +xtern/x_classifiers_liblinear_train_one_vs_all.c +xtern/x_classifiers_liblinear_defines.h +xtern/liblinear/tron.o +xtern/liblinear/linear.o +xtern/liblinear/blas/blas.a $(XTERN_H) $(XTERN_C)
	$(MEX) $(MEXFLAGS) -outdir +xtern -I+xtern +xtern/x_classifiers_liblinear_train_one_vs_all.c +xtern/liblinear/blas/blas.a +xtern/liblinear/tron.o +xtern/liblinear/linear.o $(XTERN_C) -L$(ACML_LIB_PATH) -I$(ACML_INCLUDE_PATH) $(ACML_LIBS)

+xtern/x_classifiers_liblinear_train_one_vs_one.mexa64: +xtern/x_classifiers_liblinear_train_one_vs_one.c +xtern/x_classifiers_liblinear_defines.h +xtern/liblinear/tron.o +xtern/liblinear/linear.o +xtern/liblinear/blas/blas.a $(XTERN_H) $(XTERN_C)
	$(MEX) $(MEXFLAGS) -outdir +xtern -I+xtern +xtern/x_classifiers_liblinear_train_one_vs_one.c +xtern/liblinear/blas/blas.a +xtern/liblinear/tron.o +xtern/liblinear/linear.o $(XTERN_C) -L$(ACML_LIB_PATH) -I$(ACML_INCLUDE_PATH) $(ACML_LIBS)

+xtern/x_dictionary_correlation.mexa64: +xtern/x_dictionary_correlation.c $(XTERN_H) $(XTERN_C)
	$(MEX) $(MEXFLAGS) -outdir +xtern -I+xtern +xtern/x_dictionary_correlation.c $(XTERN_C) -L$(ACML_LIB_PATH) -I$(ACML_INCLUDE_PATH) $(ACML_LIBS)

+xtern/x_dictionary_matching_pursuit.mexa64: +xtern/x_dictionary_matching_pursuit.c $(XTERN_H) $(XTERN_C)
	$(MEX) $(MEXFLAGS) -outdir +xtern -I+xtern +xtern/x_dictionary_matching_pursuit.c $(XTERN_C) -L$(ACML_LIB_PATH) -I$(ACML_INCLUDE_PATH) $(ACML_LIBS)

+xtern/x_dictionary_orthogonal_matching_pursuit.mexa64: +xtern/x_dictionary_orthogonal_matching_pursuit.c $(XTERN_H) $(XTERN_C)
	$(MEX) $(MEXFLAGS) -outdir +xtern -I+xtern +xtern/x_dictionary_orthogonal_matching_pursuit.c $(XTERN_C) -L$(ACML_LIB_PATH) -I$(ACML_INCLUDE_PATH) $(ACML_LIBS)

+xtern/x_dictionary_optimized_orthogonal_matching_pursuit.mexa64: +xtern/x_dictionary_optimized_orthogonal_matching_pursuit.c $(XTERN_H) $(XTERN_C)
	$(MEX) $(MEXFLAGS) -outdir +xtern -I+xtern +xtern/x_dictionary_optimized_orthogonal_matching_pursuit.c $(XTERN_C) -L$(ACML_LIB_PATH) -I$(ACML_INCLUDE_PATH) $(ACML_LIBS)

+xtern/x_image_recoder_code.mexa64: +xtern/x_image_recoder_code.c $(XTERN_H) $(XTERN_C)
	$(MEX) $(MEXFLAGS) -outdir +xtern -I+xtern +xtern/x_image_recoder_code.c $(XTERN_C) -L$(ACML_LIB_PATH) -I$(ACML_INCLUDE_PATH) $(ACML_LIBS)

clean:
	rm -f +xtern/test
	rm -f +xtern/*.o
	rm -f +xtern/liblinear/*.o
	rm -f +xtern/*.mexa64
	rm -f +xtern/*.mexglx
	make -C +xtern/liblinear/blas clean
