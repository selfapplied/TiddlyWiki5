#!/bin/bash

# Build script for Zeta Fractal CA WebAssembly module
# Requires Emscripten SDK: https://emscripten.org/docs/getting_started/downloads.html

set -e

echo "🔥 Building Zeta Fractal CA WebAssembly Module..."

# Check if emcc is available
if ! command -v emcc &> /dev/null; then
    echo "❌ Error: Emscripten compiler (emcc) not found"
    echo "Please install Emscripten SDK:"
    echo "  git clone https://github.com/emscripten-core/emsdk.git"
    echo "  cd emsdk"
    echo "  ./emsdk install latest"
    echo "  ./emsdk activate latest"
    echo "  source ./emsdk_env.sh"
    exit 1
fi

# Create output directory
mkdir -p wasm/build

echo "📦 Compiling C++ to WebAssembly..."

# Compile with optimization and Safari compatibility
emcc wasm/zeta_ca_kernel.cpp \
    -O3 \
    -s WASM=1 \
    -s MODULARIZE=1 \
    -s EXPORT_NAME="'ZetaCAModule'" \
    -s EXPORTED_FUNCTIONS="['_initCA', '_updateCA', '_insertPrime', '_subdivideCell', '_getCellData', '_getActiveCellCount', '_getAverageError', '_getCurrentIteration', '_getCurrentPrimeIndex']" \
    -s EXPORTED_RUNTIME_METHODS="['ccall', 'cwrap']" \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s MAXIMUM_MEMORY=64MB \
    -s TOTAL_STACK=16MB \
    -s ENVIRONMENT='web' \
    -s SINGLE_FILE=1 \
    --closure 1 \
    -o wasm/build/zeta_ca_kernel.js

echo "✅ WebAssembly module built successfully!"
echo "📄 Output: wasm/build/zeta_ca_kernel.js"
echo ""
echo "🔧 Integration notes:"
echo "  1. Copy zeta_ca_kernel.js to your TiddlyWiki plugin directory"
echo "  2. Load the module in your widget with: ZetaCAModule()"
echo "  3. Use the exported functions for CA computation"
echo ""
echo "🌀 Ready for recursive zeta descent!"