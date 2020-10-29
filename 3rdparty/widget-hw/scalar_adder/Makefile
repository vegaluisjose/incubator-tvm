VERILATOR_BIN := $(shell which verilator)
VERILATOR_INC_DIR := $(abspath $(dir $(VERILATOR_BIN))/../share/verilator/include)
TOP_NAME = "Top"
VERILOG_DIR = $(abspath .)/verilog
SRC_DIR = $(abspath .)/src
WIDGET_DIR = $(abspath ..)
TVM_DIR = $(abspath ../../..)
OUT_DIR = $(abspath .)/out

default: $(WIDGET_DIR)/libverilator.so

$(WIDGET_DIR)/libverilator.so: $(OUT_DIR)/$(TOP_NAME).cpp
	g++ \
	-std=c++14 \
	-O2 \
	-shared \
	-fPIC \
	-I$(TVM_DIR)/src/runtime/contrib/verilator \
	-I$(TVM_DIR)/include \
	-I$(TVM_DIR)/3rdparty/dlpack/include \
	-I$(VERILATOR_INC_DIR) \
	-I$(OUT_DIR) \
	$(VERILATOR_INC_DIR)/verilated.cpp \
	$(SRC_DIR)/*.cc \
	$(OUT_DIR)/*.cpp \
	-o $@

$(OUT_DIR)/$(TOP_NAME).cpp: $(VERILOG_DIR)/*.v
	$(VERILATOR_BIN) \
	-Wno-BLKANDNBLK \
	-Wno-PINMISSING \
	-Wno-STMTDLY \
	-Wno-WIDTH \
	-Wno-UNOPTFLAT \
	--cc \
	--prefix $(TOP_NAME) \
	--top-module "driver" \
	--Mdir $(OUT_DIR) \
	$^

clean:
	-rm -rf $(OUT_DIR)
