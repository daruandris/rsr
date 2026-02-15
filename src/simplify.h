#ifndef SIMPLIFY_H
#define SIMPLIFY_H

extern "C" {
    void simplify_symengine_cpp(const char* input, char* output, int max_len);
}

#endif