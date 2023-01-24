#ifndef UTILITY_IPP
#define UTILITY_IPP

#include <cmath>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

template <typename T> std::string vector_to_string(std::vector<T> vec) {
    std::ostringstream oss;
    if (!vec.empty()) {
        std::copy(vec.begin(), vec.end() - 1, std::ostream_iterator<T>(oss, ","));
        oss << vec.back();
    }

    return "(" + oss.str() + ")";
}

template <typename T> void print_vector(std::vector<T> vec) { std::cout << vector_to_string(vec) << std::endl; }

template <typename T> bool are_same_vectors(std::vector<T> a, std::vector<T> b) {
    return std::equal(a.begin(), a.end(), b.begin());
}

#endif
