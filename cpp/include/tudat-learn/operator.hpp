/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_OPERATOR_HPP
#define TUDAT_LEARN_OPERATOR_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "tudat-learn/types.hpp"

namespace tudat_learn
{

template <typename Datum_t>
class Operator {
  protected:
    Operator() { }

    /* BEGIN BASIC ARITHMETIC OPERATIONS */

    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value || is_eigen<T>::value,
    T >::type              operator_difference(T lhs, const T &rhs) const { return lhs -= rhs; }

    // implements an operator difference for vector<arithmetic> types
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    std::vector<T> >::type operator_difference(std::vector<T> lhs, const std::vector<T> &rhs) const {
      for(int i = 0; i < lhs.size(); ++i)
        lhs.at(i) -= rhs.at(i);
      return lhs;
    }


    // implements an operator sum for arithmetic and eigen types
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value || is_eigen<T>::value,
    T >::type              operator_addition(T lhs, const T &rhs) const { return lhs += rhs; }

    // implements an operator sum for vector<arithmetic> types
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    std::vector<T> >::type operator_addition(std::vector<T> lhs, const std::vector<T> &rhs) const {
      for(int i = 0; i < lhs.size(); ++i)
        lhs.at(i) += rhs.at(i);
      return lhs;
    }


    // implements a multiplication for arithmetic types
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    T>::type               operator_elementwise_multiplication(T lhs, const T &rhs) const { return lhs *= rhs; }

    // implements an element-wise multiplication for eigen types
    template <typename T>
    typename std::enable_if<           is_eigen<T>::value,
    T>::type               operator_elementwise_multiplication(T lhs, const T &rhs) const { return lhs.array() *= rhs.array(); }

    // implements an element-wise multiplication for vector<arithmetic> types
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    std::vector<T> >::type operator_elementwise_multiplication(std::vector<T> lhs, const std::vector<T> &rhs) const { 
      std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), lhs.begin(), std::multiplies<T>()); 
      return lhs;
    }


    // implements a division for arithmetic types
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    T>::type               operator_elementwise_division(T lhs, const T &rhs) const { return lhs /= rhs; }

    // implements an element-wise division for eigen types
    template <typename T>
    typename std::enable_if<           is_eigen<T>::value,
    T>::type               operator_elementwise_division(T lhs, const T &rhs) const { return lhs.array() /= rhs.array(); }

    // implements an element-wise division for vector<arithmetic> types
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    std::vector<T> >::type operator_elementwise_division(std::vector<T> lhs, const std::vector<T> &rhs) const { 
      std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), lhs.begin(), std::divides<T>()); 
      return lhs;
    }


    // implements an addition for arithmetic types
    template <typename T, typename U>
    typename std::enable_if< std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,
    T>::type               operator_scalar_addition(T lhs, U rhs) const { return lhs += rhs; }

    // implements an addition of a scalar for eigen types
    template <typename T, typename U>
    typename std::enable_if<           is_eigen<T>::value && std::is_arithmetic<U>::value,
    T>::type               operator_scalar_addition(T lhs, U rhs) const { return lhs.array() += rhs; }

    // implements an addition of a scalar for vector<arithmetic> types
    template <typename T, typename U>
    typename std::enable_if< std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,
    std::vector<T> >::type operator_scalar_addition(std::vector<T> lhs, U rhs) const { 
      for(auto &it : lhs)
        it += rhs;
      return lhs;
    }

    // implements an subtraction for arithmetic types
    template <typename T, typename U>
    typename std::enable_if< std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,
    T>::type               operator_scalar_subtraction(T lhs, U rhs) const { return lhs -= rhs; }

    // implements an subtraction of a scalar for eigen types
    template <typename T, typename U>
    typename std::enable_if<           is_eigen<T>::value && std::is_arithmetic<U>::value,
    T>::type               operator_scalar_subtraction(T lhs, U rhs) const { return lhs.array() -= rhs; }

    // implements an subtraction of a scalar for vector<arithmetic> types
    template <typename T, typename U>
    typename std::enable_if< std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,
    std::vector<T> >::type operator_scalar_subtraction(std::vector<T> lhs, U rhs) const { 
      for(auto &it : lhs)
        it -= rhs;
      return lhs;
    }


    // implements a division for arithmetic types
    template <typename T, typename U>
    typename std::enable_if< std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,
    T>::type               operator_scalar_division(T lhs, U rhs) const { return lhs /= rhs; }

    // implements a division by a scalar for eigen types
    template <typename T, typename U>
    typename std::enable_if<           is_eigen<T>::value && std::is_arithmetic<U>::value,
    T>::type               operator_scalar_division(T lhs, U rhs) const { return lhs.array() /= rhs; }

    // implements a division by a scalar for vector<arithmetic> types
    template <typename T, typename U>
    typename std::enable_if< std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,
    std::vector<T> >::type operator_scalar_division(std::vector<T> lhs, U rhs) const { 
      for(auto &it : lhs)
        it /= rhs;
      return lhs;
    }

    /* END BASIC ARITHMETIC OPERATIONS */


    /* BEGIN OTHER ARITHMETIC OPERATIONS */

    // implements square root for arithmetic types
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value, 
    T>::type             square_root(const T &arg) const { return std::sqrt(arg); }

    // implements element-wise square root for eigen types
    template <typename T>
    typename std::enable_if<           is_eigen<T>::value, 
    T>::type             square_root(const T &arg) const { return Eigen::sqrt(arg.array()); }

    // implements element-wise square root for vector types
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value, 
    std::vector<T>>::type square_root(std::vector<T> arg) const { 
      for(auto &it: arg)
        it = std::sqrt(it);
      return arg; 
    }



    /* END OTHER ARITHMETIC OPERATIONS */

    /* BEGIN COMPARISONS */

    // implements an operator <= for arithmetic types
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    bool>::type operator_leq(const T &lhs, const T &rhs) const { return lhs <= rhs; }

    // implements an operator (lhs <= rhs).any() for eigen types
    template <typename T>
    typename std::enable_if<           is_eigen<T>::value,
    bool>::type operator_leq(const T &lhs, const T &rhs) const { return (lhs.array() <= rhs.array()).any(); }

    // implements an operator (lhs <= rhs).any() for vector<arithmetic> types
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    bool>::type operator_leq(const std::vector<T> &lhs, const std::vector<T> &rhs) const { 
      for(int i = 0; i < lhs.size(); ++i)
        if(lhs.at(i) >= rhs.at(i))
          return false;

      return true;
    }

    // implements an element-wise max for arithmetic types
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    T>::type               elementwise_max(const T &lhs, const T &rhs) const { return (lhs > rhs) ? lhs : rhs; }

    // implements an element-wise max for eigen types
    template <typename T>
    typename std::enable_if<            is_eigen<T>::value,
    T>::type               elementwise_max(const T &lhs, const T &rhs) const { return lhs.array().max(rhs.array()); }

    // implements an element-wise max for vector<arithmetic> types
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    std::vector<T> >::type elementwise_max(std::vector<T> &lhs, const std::vector<T> &rhs) const { 
      std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), lhs.begin(),
        [](const T lhs_element, const T rhs_element){ return (lhs_element > rhs_element) ? lhs_element : rhs_element; } );
      return lhs;
    }

    // implements an element-wise min for arithmetic types
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    T>::type               elementwise_min(const T &lhs, const T &rhs) const { return (lhs < rhs) ? lhs : rhs; }

    // implements an element-wise min for eigen types
    template <typename T>
    typename std::enable_if<            is_eigen<T>::value,
    T>::type               elementwise_min(const T &lhs, const T &rhs) const { return lhs.array().min(rhs.array()); }

    // implements an element-wise min for vector<arithmetic> types
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    std::vector<T> >::type elementwise_min(std::vector<T> &lhs, const std::vector<T> &rhs) const { 
      std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), lhs.begin(),
        [](const T lhs_element, const T rhs_element){ return (lhs_element < rhs_element) ? lhs_element : rhs_element; } );
      return lhs;
    }

    // check whether two elements have the same dimensions for arithmetic types
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    bool>::type check_dimensions(const T &lhs, const T &rhs) const { return true; }

    // check whether two elements have the same dimensions for eigen types
    template <typename T>
    typename std::enable_if<            is_eigen<T>::value,
    bool>::type check_dimensions(const T &lhs, const T &rhs) const { return (lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols()); }

    // check whether two elements have the same dimensions for vector<arithmetic> types
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    bool>::type check_dimensions(const std::vector<T> &lhs, const std::vector<T> &rhs) const { return (lhs.size() == rhs.size()); }



    /* END COMPARISONS */

    /* BEGIN PRINTING OPERATIONS */

    // prints an arithmetic or eigen type
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value || is_eigen<T>::value,
    void >::type print_datum_t(const T &to_print) const { std::cout << to_print << std::endl; }

    // prints a vector<arithmetic> type
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    void >::type print_datum_t(const std::vector<T> &to_print) const {
      for(const auto &it: to_print)
        std::cout << it << ", ";
      std::cout << "\n" << std::endl;
    }


    // prints a vector of arithmetic types
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    void >::type print_vector_datum_t(const std::vector<T> &to_print) const {
      for(const auto &it: to_print)
        std::cout << it << ", ";
      std::cout << "\n";
    }

    // prints a vector of eigen types
    template <typename T>
    typename std::enable_if<           is_eigen<T>::value,
    void >::type print_vector_datum_t(const std::vector<T> &to_print) const {
      for(const auto &it: to_print)
        std::cout << it << "\n" << std::endl;
    }

    // prints a vector of vector<arithmetic> types
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    void >::type print_vector_datum_t(const std::vector<std::vector<T>> &to_print) const {
      for(const auto &it: to_print) {
        for(const auto &itt: it)
          std::cout << itt << ", ";

        std::cout << "\n" << std::endl;
      }
    }

    /* END PRINTING OPERATIONS */

    /* BEGIN OTHER OPERATIONS */

    // implements a method that retrieves the dimension if the datum is of arithmetic type
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    int>::type get_dimensions(const T &datum) const
    { return 1; }

    // implements a method that retrieves the dimension if the datum is of eigen type
    template <typename T>
    typename std::enable_if<            is_eigen<T>::value,
    int>::type get_dimensions(const T &datum) const
    { return datum.rows() * datum.cols(); }

    // implements a method that retrieves the dimension if the datum is of arithmetic type
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    int>::type get_dimensions(const std::vector<T> &datum) const
    { return datum.size(); }

    /* END OTHER OPERATIONS */
};

} // namespace tudat_learn

#endif // TUDAT_LEARN_OPERATOR_HPP