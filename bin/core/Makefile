CXX ?= g++
CC ?= gcc
MEX = /opt/MatLAB/R2012a/bin/mex
INCLUDE_PATH = +xtern/32bit/include
LIB_PATH = +xtern/32bit/lib
LIBLINEAR_OBJ = +xtern/32bit/lib/tron.o +xtern/32bit/lib/linear.o
# MEX = /opt/local-ubuntu/MATLAB/R2011a/bin/mex
# INCLUDE_PATH = +xtern/64bit/include
# LIB_PATH = +xtern/64bit/lib
# LIBLINEAR_OBJ = +xtern/64bit/lib/tron.o +xtern/64bit/lib/linear.o
LIBS = -lgsl -lcblas -lacml
CFLAGS = -fstrict-aliasing -Wstrict-aliasing -g -Wall -Wconversion -fPIC -I$(INCLUDE_PATH) -L$(LIB_PATH) -D_GNU_SOURCE
MEXFLAGS = -g CC\#$(CXX) CXX\#$(CXX) CFLAGS\#"$(CFLAGS)" CXXFLAGS\#"$(CFLAGS)" -largeArrayDims
XTERN_BASE_H = +xtern/base_defines.h +xtern/latools.h +xtern/coding_methods.h +xtern/image_coder.h +xtern/task_control.h
XTERN_BASE_C = +xtern/latools.c +xtern/coding_methods.c +xtern/image_coder.c +xtern/task_control.c
XTERN_H = +xtern/x_mex_interface.h $(XTERN_BASE_H)
XTERN_C = +xtern/x_mex_interface.c $(XTERN_BASE_C)

all: +xtern/test +xtern/x_classifiers_liblinear_classify.mexa64 +xtern/x_classifiers_liblinear_train_one_vs_all.mexa64 +xtern/x_classifiers_liblinear_train_one_vs_one.mexa64 +xtern/x_dictionary_correlation.mexa64 +xtern/x_dictionary_matching_pursuit.mexa64 +xtern/x_dictionary_orthogonal_matching_pursuit.mexa64 +xtern/x_dictionary_optimized_orthogonal_matching_pursuit.mexa64 +xtern/x_dictionary_sparse_net.mexa64 +xtern/x_image_recoder_code.mexa64

+xtern/test: +xtern/test.c $(XTERN_BASE_C) $(XTERN_BASE_H)
	gcc $(CFLAGS) -o +xtern/test +xtern/test.c $(XTERN_BASE_C) -lpthread -lm $(LIBS)

+xtern/x_classifiers_liblinear_classify.mexa64: +xtern/x_classifiers_liblinear_classify.c +xtern/x_classifiers_liblinear_defines.h $(XTERN_H) $(XTERN_C)
	$(MEX) $(MEXFLAGS) -outdir +xtern +xtern/x_classifiers_liblinear_classify.c $(LIBLINEAR_OBJ) $(XTERN_C) -I$(INCLUDE_PATH) -L$(LIB_PATH) $(LIBS)

+xtern/x_classifiers_liblinear_train_one_vs_all.mexa64: +xtern/x_classifiers_liblinear_train_one_vs_all.c +xtern/x_classifiers_liblinear_defines.h $(XTERN_H) $(XTERN_C)
	$(MEX) $(MEXFLAGS) -outdir +xtern +xtern/x_classifiers_liblinear_train_one_vs_all.c $(LIBLINEAR_OBJ) $(XTERN_C) -I$(INCLUDE_PATH) -L$(LIB_PATH) $(LIBS)

+xtern/x_classifiers_liblinear_train_one_vs_one.mexa64: +xtern/x_classifiers_liblinear_train_one_vs_one.c +xtern/x_classifiers_liblinear_defines.h  $(XTERN_H) $(XTERN_C)
	$(MEX) $(MEXFLAGS) -outdir +xtern +xtern/x_classifiers_liblinear_train_one_vs_one.c $(LIBLINEAR_OBJ) $(XTERN_C) -I$(INCLUDE_PATH) -L$(LIB_PATH) $(LIBS)

+xtern/x_dictionary_correlation.mexa64: +xtern/x_dictionary_correlation.c $(XTERN_H) $(XTERN_C)
	$(MEX) $(MEXFLAGS) -outdir +xtern +xtern/x_dictionary_correlation.c $(XTERN_C) -I$(INCLUDE_PATH) -L$(LIB_PATH) $(LIBS)

+xtern/x_dictionary_matching_pursuit.mexa64: +xtern/x_dictionary_matching_pursuit.c $(XTERN_H) $(XTERN_C)
	$(MEX) $(MEXFLAGS) -outdir +xtern +xtern/x_dictionary_matching_pursuit.c $(XTERN_C) -I$(INCLUDE_PATH) -L$(LIB_PATH) $(LIBS)

+xtern/x_dictionary_orthogonal_matching_pursuit.mexa64: +xtern/x_dictionary_orthogonal_matching_pursuit.c $(XTERN_H) $(XTERN_C)
	$(MEX) $(MEXFLAGS) -outdir +xtern +xtern/x_dictionary_orthogonal_matching_pursuit.c $(XTERN_C) -I$(INCLUDE_PATH) -L$(LIB_PATH) $(LIBS)

+xtern/x_dictionary_optimized_orthogonal_matching_pursuit.mexa64: +xtern/x_dictionary_optimized_orthogonal_matching_pursuit.c $(XTERN_H) $(XTERN_C)
	$(MEX) $(MEXFLAGS) -outdir +xtern +xtern/x_dictionary_optimized_orthogonal_matching_pursuit.c $(XTERN_C) -I$(INCLUDE_PATH) -L$(LIB_PATH) $(LIBS)

+xtern/x_dictionary_sparse_net.mexa64: +xtern/x_dictionary_sparse_net.c $(XTERN_H) $(XTERN_C)
	$(MEX) $(MEXFLAGS) -outdir +xtern +xtern/x_dictionary_sparse_net.c $(XTERN_C) -I$(INCLUDE_PATH) -L$(LIB_PATH) $(LIBS)

+xtern/x_image_recoder_code.mexa64: +xtern/x_image_recoder_code.c $(XTERN_H) $(XTERN_C)
	$(MEX) $(MEXFLAGS) -outdir +xtern +xtern/x_image_recoder_code.c $(XTERN_C) -I$(INCLUDE_PATH) -L$(LIB_PATH) $(LIBS)

clean:
	rm -f +xtern/test
	rm -f +xtern/*.mexa64
	rm -f +xtern/*.mexglx
