# ------ Test parameters ------
SIZE  := 512
DEPTH := 128
RUNS  := 5
TX    := 100
TY    := -100
ANG   := 25.34
TASK  ?= WARP

SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin

# HIP flags for Ubuntu
HIPCC    := hipcc
HIPFLAGS := -O3 -std=c++17

# Include paths (modify if necessary)
HIPFLAGS += -I./externals -I./src/utils -I./src/hip_kernels -DCPU_CHECK

# Target: HIP test executable
TARGET := $(BIN_DIR)/test_rigid_warp_hip

# Sources: include main and all other .cpp files in src and its subdirectories
SOURCES := $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/utils/*.cpp) $(wildcard $(SRC_DIR)/hip_kernels/*.cpp)

# Objects: transform sources into .o files, maintaining directory structure
OBJ := $(patsubst $(SRC_DIR)/%, $(OBJ_DIR)/%, $(SOURCES:.cpp=.o))

# Default rule: compile the target
all: $(TARGET)

# Rule to compile each source into object
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

# Rule for linking the executable
$(TARGET): $(OBJ)
	@mkdir -p $(BIN_DIR)
	$(HIPCC) $(HIPFLAGS) $(OBJ) -o $(TARGET)

# Rule to run the test with default parameters
run: $(TARGET)
	@echo "Running $(TARGET) with the following parameters:"
	@echo ">>> $(SIZE)x$(SIZE)x$(DEPTH) | $(RUNS) run | (tx=$(TX)px, ty=$(TY)px, ang=$(ANG)d)"
	$(TARGET) --task $(TASK) -s $(SIZE) -d $(DEPTH) -r $(RUNS) -x $(TX) -y $(TY) -a $(ANG)

# Clean objects and executable
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all run clean
