#ifndef SIMPLIFY_H
#define SIMPLIFY_H

#include <stddef.h>

extern "C" {
    void simplify_symengine_cpp(const char* input, char* output, size_t max_len);
}

#endif