#ifndef SYMENGINE_CONFIG_CLING_HPP
#define SYMENGINE_CONFIG_CLING_HPP

#ifdef __CLING__

#pragma cling add_library_path("C:/Users/darua/Documents/rust_learning/rsr/target/release/build/rsr-fc2994bd84bdf16d/out/lib")
#pragma cling load("symengine")

#elif defined(__EMSCRIPTEN__) && defined(__CLANG_REPL__)

#include <clang/Interpreter/CppInterOp.h>
static bool _symengine_loaded = []() {
    Cpp::LoadLibrary("/lib/symengine.dll", false);
    return true;
}();

#endif

#endif // SYMENGINE_CONFIG_CLING_HPP
