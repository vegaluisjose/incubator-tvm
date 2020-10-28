TVM_DIR = $(abspath ../..)
SRC_DIR = $(abspath .)/scalar_adder/src

libverilator.so:
	g++ -I$(TVM_DIR)/src/runtime/contrib/verilator \
	-I$(TVM_DIR)/include \
	-I$(TVM_DIR)/3rdparty/dlpack/include \
	-shared -fPIC \
	$(SRC_DIR)/*.cc \
	-o $@	
