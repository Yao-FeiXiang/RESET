CXX := g++
CXXFLAGS := -O3 -std=c++17 -Wall -Wextra -Wno-unused-parameter -Iinclude
LDFLAGS :=

BIN_DIR := bin
BINS := $(BIN_DIR)/extract_pairs $(BIN_DIR)/build_csr_partition

all: $(BINS)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(BIN_DIR)/extract_pairs: extract_pairs.cpp include/common.h include/varint.h include/bcsr_reader.h | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ extract_pairs.cpp $(LDFLAGS)

$(BIN_DIR)/build_csr_partition: build_csr_partition.cpp include/common.h | $(BIN_DIR)
	$(CXX) $(CXXFLAGS) -o $@ build_csr_partition.cpp $(LDFLAGS)

clean:
	rm -rf $(BIN_DIR)
