.PHONY: clean dirs

BIN=bin
SRC=src
TEST=test
OBJ=obj
LIBS=-lgtest -lpthread -lgsl -lgslcblas -lm

all: dirs $(BIN)/unit_test

$(BIN)/unit_test: $(TEST)/unit_test.cpp $(TEST)/unit_test.h \
$(SRC)/neural_network.h $(SRC)/matrix.h $(SRC)/matrix_multiplication.h \
$(OBJ)/activation_function.o $(OBJ)/fully_connected_layer.o $(OBJ)/convolution.o \
$(OBJ)/random.o | $(BIN)
	g++ -Wall -g -std=c++11 $< \
	$(OBJ)/activation_function.o \
	$(OBJ)/fully_connected_layer.o \
	$(OBJ)/convolution.o \
	$(OBJ)/random.o \
	$(LIBS) -o $@

$(OBJ)/activation_function.o: $(SRC)/activation_function.cpp $(SRC)/activation_function.h | $(OBJ)
	g++ -Wall -std=c++11 -c $< -o $@ $(LIBS)

$(OBJ)/fully_connected_layer.o: $(SRC)/fully_connected_layer.cpp $(SRC)/fully_connected_layer.h | $(OBJ)
	g++ -Wall -std=c++11 -c $< -o $@ $(LIBS)

$(OBJ)/convolution.o: $(SRC)/convolution.cpp $(SRC)/convolution.h | $(OBJ)
	g++ -Wall -std=c++11 -c $< -o $@ $(LIBS)

$(OBJ)/random.o: $(SRC)/random.cpp $(SRC)/random.h | $(OBJ)
	g++ -Wall -std=c++11 -c $< -o $@ $(LIBS)

dirs:
	mkdir -p $(SRC) $(TEST) $(BIN) $(OBJ)

clean:
	rm -rf $(BIN) $(OBJ)

stat:
	wc src/* test/*