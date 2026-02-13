# ------ Parametri di test ------
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

# Impostazioni HIP per Ubuntu
HIPCC    := hipcc
HIPFLAGS := -O3 -std=c++17

# Include paths (modifica se necessario)
HIPFLAGS += -I./externals -I./src/utils -I./src/hip_kernels -DCPU_CHECK

# Target: eseguibile del test HIP
TARGET := $(BIN_DIR)/test_rigid_warp_hip

# Sorgenti: includi il main e tutti gli altri file .cpp presenti in src e nelle sottocartelle
SOURCES := $(wildcard $(SRC_DIR)/*.cpp) $(wildcard $(SRC_DIR)/utils/*.cpp) $(wildcard $(SRC_DIR)/hip_kernels/*.cpp)

# Oggetti: trasformiamo i sorgenti in file .o, mantenendo la struttura di directory
OBJ := $(patsubst $(SRC_DIR)/%, $(OBJ_DIR)/%, $(SOURCES:.cpp=.o))

# Regola di default: compila il target
all: $(TARGET)

# Regola per compilare ogni sorgente in oggetto
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(dir $@)
	$(HIPCC) $(HIPFLAGS) -c $< -o $@

# Regola per il link dell'eseguibile
$(TARGET): $(OBJ)
	@mkdir -p $(BIN_DIR)
	$(HIPCC) $(HIPFLAGS) $(OBJ) -o $(TARGET)

# Regola per eseguire il test con i parametri predefiniti
run: $(TARGET)
	@echo "Esecuzione di $(TARGET) con i seguenti parametri:"
	@echo ">>> $(SIZE)x$(SIZE)x$(DEPTH) | $(RUNS) run | (tx=$(TX)px, ty=$(TY)px, ang=$(ANG)d)"
	$(TARGET) --task $(TASK) -s $(SIZE) -d $(DEPTH) -r $(RUNS) -x $(TX) -y $(TY) -a $(ANG)

# Pulizia degli oggetti e dell'eseguibile
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all run clean
