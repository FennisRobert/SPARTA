# Builds the SPARTA shared library, the C and C++ examples, and the API checks.
#
#   make run-c      build everything and run the C example
#   make run-cpp    build everything and run the C++ example
#   make test       build and run the API checks
#   make package    assemble a redistributable SDK under dist/
#   make clean      remove the build output

VERSION  := 0.3.1
# DIST, TARGET and CARGO_FLAGS may be overridden.
DIST     ?= dist/sparta-$(VERSION)-macos-arm64
TARGET   ?= $(abspath ../../target/release)
CARGO_FLAGS ?=
LIBRARY  := $(TARGET)/libsparta.dylib
BUILD    := build

CFLAGS   := -std=c11 -O2 -Wall -Wextra -Iinclude
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Iinclude
LDFLAGS  := -L$(TARGET) -lsparta -Wl,-rpath,$(TARGET)

.PHONY: all library run-c run-cpp test package clean

all: $(BUILD)/laplace-c $(BUILD)/laplace-cpp $(BUILD)/api_test

# ---- the library ------------------------------------------------------------

library:
	cargo build --release -p sparta-capi $(CARGO_FLAGS)

$(LIBRARY): library

# ---- programs built against it ----------------------------------------------

$(BUILD):
	mkdir -p $(BUILD)

$(BUILD)/laplace-c: examples/laplace.c include/sparta.h $(LIBRARY) | $(BUILD)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

$(BUILD)/laplace-cpp: examples/laplace.cpp include/sparta.hpp include/sparta.h $(LIBRARY) | $(BUILD)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

$(BUILD)/api_test: tests/api_test.c include/sparta.h $(LIBRARY) | $(BUILD)
	$(CC) $(CFLAGS) $< -o $@ $(LDFLAGS)

run-c: $(BUILD)/laplace-c
	@echo && $<

run-cpp: $(BUILD)/laplace-cpp
	@echo && $<

test: $(BUILD)/api_test
	@echo && $<

# ---- what a customer receives -----------------------------------------------

# The install name is rewritten to @rpath so an application can embed the
# library wherever its own layout puts it.
#
# Sign and notarize before sending it out, with your own Developer ID:
#     codesign -s "Developer ID Application: ..." -o runtime --timestamp \
#         $(DIST)/lib/libsparta.dylib
package: library
	rm -rf $(DIST)
	mkdir -p $(DIST)/include $(DIST)/lib $(DIST)/examples
	cp include/sparta.h include/sparta.hpp $(DIST)/include/
	cp $(LIBRARY) $(DIST)/lib/
	install_name_tool -id @rpath/libsparta.dylib $(DIST)/lib/libsparta.dylib
	cp examples/laplace.c examples/laplace.cpp $(DIST)/examples/
	cp ../../THIRD-PARTY-NOTICES README.md $(DIST)/
	@echo && echo "packaged $(DIST)"

clean:
	rm -rf $(BUILD) dist
