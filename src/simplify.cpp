#include "simplify.h"
#include <symengine/expression.h>
#include <symengine/simplify.h>
#include <string>
#include <cstring>

extern "C" {
    void simplify_symengine_cpp(const char* input, char* output, int max_len) {
        try {
            SymEngine::Expression expr(input);
            std::string simplified = SymEngine::expand(expr.get_basic())->__str__();
            
            strncpy_s(output, max_len, simplified.c_str(), _TRUNCATE);
        }
        catch(...) {
            strncpy_s(output, max_len, input, _TRUNCATE);
        }
    }
}