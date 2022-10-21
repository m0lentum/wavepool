# program executable, program folder, and library folder
TARGET = run
BUILD = build
LIB := ../gfd/GFD

# c++ compiler and compiler flags
CXX = mpicxx
CXXFLAGS := -g -Wall -Wextra -pedantic -std=c++11
N = 1

# commands for insert and delete folder
MKDIR_P ?= mkdir -p
RMDIR_R ?= $(RM) -r

# source files for the program
SOURCES := Main.cpp $(wildcard $(LIB)/Types/*.cpp) $(wildcard $(LIB)/Mesh/*.cpp) $(wildcard $(LIB)/Output/*.cpp) $(wildcard $(LIB)/BlockDec/*.cpp) $(wildcard $(LIB)/Discrete/*.cpp)
OBJECTS := $(addprefix $(BUILD)/,$(patsubst %.cpp,%.o,$(patsubst $(LIB)/%,%,$(SOURCES))))

# general part of makefile
all: $(BUILD)/$(TARGET)
$(BUILD)/$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@
$(BUILD)/%.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@
$(BUILD)/%.o: $(LIB)/%.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

.PHONY: clean
clean:
	$(RMDIR_R) $(BUILD)
run: $(BUILD)/$(TARGET)
	cd $(BUILD); ./$(TARGET); cd ../
mpirun: $(BUILD)/$(TARGET)
	cd $(BUILD); mpirun -np $(N) ./$(TARGET); cd ../

