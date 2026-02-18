#include "simplify.h"
#include <symengine/expression.h>
#include <symengine/simplify.h>
#include <symengine/real_double.h>
#include <symengine/add.h>
#include <symengine/mul.h>
#include <symengine/pow.h>
#include <symengine/functions.h>
#include <symengine/visitor.h>
#include <cmath>
#include <vector>
#include <string>
#include <cstdio>

using namespace SymEngine;

RCP<const Basic> round_floats(const RCP<const Basic> &x) {
    if (is_a<RealDouble>(*x)) {
        double val = down_cast<const RealDouble &>(*x).as_double();
        double rounded = std::round(val * 10000.0) / 10000.0;
        if (std::abs(rounded - std::round(rounded)) < 1e-9) {
            return integer(static_cast<long>(std::round(rounded)));
        }
        return real_double(rounded);
    }
    vec_basic args = x->get_args();
    if (args.empty()) {
        return x;
    }
    vec_basic new_args;
    bool changed = false;
    new_args.reserve(args.size());

    for (const auto &arg : args) {
        auto new_arg = round_floats(arg);
        new_args.push_back(new_arg);
        if (new_arg != arg) {
            changed = true;
        }
    }
    if (!changed) {
        return x;
    }
    if (is_a<Add>(*x)) return add(new_args);
    if (is_a<Mul>(*x)) return mul(new_args);
    if (is_a<Pow>(*x)) return pow(new_args[0], new_args[1]);
    if (is_a<Sin>(*x)) return sin(new_args[0]);
    if (is_a<Cos>(*x)) return cos(new_args[0]);
    if (is_a<Log>(*x)) return log(new_args[0]); 
    return x;
}

extern "C" {
    void simplify_symengine_cpp(const char* input, char* output, size_t max_len) {
        if (max_len > 0) output[0] = '\0';
        try {
            Expression expr(input);
            auto expanded = expand(expr.get_basic());
            auto rounded = round_floats(expanded);
            std::string result = rounded->__str__();
            snprintf(output, max_len, "%s", result.c_str());
        }
        catch (...) {
            snprintf(output, max_len, "%s", input);
        }
    }
}