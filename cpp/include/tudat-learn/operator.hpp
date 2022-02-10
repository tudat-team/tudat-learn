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

/**
 * @brief Implements multiple operations for different types, using type traits. \n
 * Ensures that the classes that inherit from Operator have standardised operations that can be used with multiple types,
 * such as scalars, Eigen::Matrix, or std::vector. \n
 * Even though it takes Datum_t as a parameter, most of the operations can be called for the Label_t, as the types are deduced
 * from the arguments of the functions themselves. However, there is an exception, when the specialization that is chosen by
 * the compiler does not depend on the its arguments, but rather on the output type, which is usually the Datum_t type.
 * 
 * @tparam Datum_t The type of a single feature vector. Can be a simple scalar, an std::vector<T>, an Eigen::Matrix, ...
 */
template <typename Datum_t>
class Operator {
  protected:
    /**
     * @brief Protected default constructor, restricting initialization to be only possible within the classes that inherit from 
     * Operator
     * 
     */
    Operator() = default;

    /* BEGIN BASIC ARITHMETIC OPERATIONS */

    /**
     * @brief Implements a difference operation between lhs and rhs for arithmetic and Eigen types. \n 
     * Returns the equivalent to (lhs - rhs).
     * 
     * @tparam T Type of the parameters being used in the operation.
     * @param lhs A copy of the first argument is used to return the result.
     * @param rhs A constant reference to the second argument.
     * @return std::enable_if< std::is_arithmetic<T>::value || is_eigen<T>::value,
     * T >::type The difference between lhs and rhs, if the T is of an arithmetic or Eigen type.
     */
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value || is_eigen<T>::value,
    T >::type              operator_difference(T lhs, const T &rhs) const { return lhs -= rhs; }

    /**
     * @brief Implements a difference operation between lhs and rhs for std::vector<arithmetic> types. \n
     * Returns the equivalent to (lhs - rhs), implementing it by iterating through every element of the input vectors.
     * 
     * @tparam T Type of the elements of the vectors being used in the operation.
     * @param lhs A copy of the first argument is used to return the result.
     * @param rhs A constant reference to the second argument.
     * @return std::enable_if< std::is_arithmetic<T>::value,
     * std::vector<T> >::type The difference between lhs and rhs, if the inputs are of a vector<arithmetic> type.
     */
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    std::vector<T> >::type operator_difference(std::vector<T> lhs, const std::vector<T> &rhs) const {
      for(std::size_t i = 0; i < lhs.size(); ++i)
        lhs.at(i) -= rhs.at(i);
      return lhs;
    }


    /**
     * @brief Implements an addition operation of lhs and rhs for arithmetic and Eigen types. \n 
     * Returns the equivalent to (lhs + rhs).
     * 
     * @tparam T Type of the parameters being used in the operation.
     * @param lhs A copy of the first argument is used to return the result.
     * @param rhs A constant reference to the second argument.
     * @return std::enable_if< std::is_arithmetic<T>::value || is_eigen<T>::value,
     * T >::type The addition of lhs and rhs, if the T is of an arithmetic or Eigen type.
     */
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value || is_eigen<T>::value,
    T >::type              operator_addition(T lhs, const T &rhs) const { return lhs += rhs; }

    /**
     * @brief Implements an addition operation of lhs and rhs for std::vector<arithmetic> types. \n
     * Returns the equivalent to (lhs + rhs), implementing it by iterating through every element of the input vectors.
     * 
     * @tparam T Type of the elements of the vectors being used in the operation.
     * @param lhs A copy of the first argument is used to return the result.
     * @param rhs A constant reference to the second argument.
     * @return std::enable_if< std::is_arithmetic<T>::value,
     * std::vector<T> >::type The addition of lhs and rhs, if the inputs are of a vector<arithmetic> type.
     */
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    std::vector<T> >::type operator_addition(std::vector<T> lhs, const std::vector<T> &rhs) const {
      for(decltype(lhs.size()) i = 0; i < lhs.size(); ++i)
        lhs.at(i) += rhs.at(i);
      return lhs;
    }

    /**
     * @brief Implements a multiplication operation between lhs and rhs for arithmetic types. \n 
     * Returns the equivalent to (lhs * rhs).
     * 
     * @tparam T Type of the parameters being used in the operation.
     * @param lhs A copy of the first argument is used to return the result.
     * @param rhs A constant reference to the second argument.
     * @return std::enable_if< std::is_arithmetic<T>::value,
     * T>::type The multiplication of lhs and rhs, if T is of an arithmetic type. 
     */
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    T>::type               operator_elementwise_multiplication(T lhs, const T &rhs) const { return lhs *= rhs; }

    /**
     * @brief Implements an element-wise multiplication operation between lhs and rhs for Eigen types. \n
     * 
     * @tparam T Type of the parameters being used in the operation.
     * @param lhs A copy of the first argument is used to return the result.
     * @param rhs A constant reference to the second argument.
     * @return std::enable_if<           is_eigen<T>::value,
     * T>::type The element-wise multiplication of lhs and rhs, if T is of an Eigen type.
     */
    template <typename T>
    typename std::enable_if<           is_eigen<T>::value,
    T>::type               operator_elementwise_multiplication(T lhs, const T &rhs) const { return lhs.array() *= rhs.array(); }

    /**
     * @brief Implements an element-wise multiplication operation between lhs and rhs for vector<arithmetic> types. \n
     * 
     * @tparam T Type of the elements of the vectors being used in the operation.
     * @param lhs A copy of the first argument is used to return the result.
     * @param rhs A constant reference to the second argument.
     * @return std::enable_if< std::is_arithmetic<T>::value,
     * std::vector<T> >::type The element-wise multiplication of lhs and rhs, if the inputs are of a vector<arithmetic> type.
     */
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    std::vector<T> >::type operator_elementwise_multiplication(std::vector<T> lhs, const std::vector<T> &rhs) const { 
      std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), lhs.begin(), std::multiplies<T>()); 
      return lhs;
    }


    /**
     * @brief Implements a division operation of lhs by rhs for arithmetic types.
     * Returns the equivalent to (lhs / rhs).
     * 
     * @tparam T Type of the parameters being used in the operation.
     * @param lhs A copy of the first argument is used to return the result.
     * @param rhs A constant reference to the second argument.
     * @return std::enable_if< std::is_arithmetic<T>::value,
     * T>::type The division of lhs by rhs, if T is of an arithmetic type. 
     */
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    T>::type               operator_elementwise_division(T lhs, const T &rhs) const { return lhs /= rhs; }

    /**
     * @brief Implements an element-wise division operation of lhs by rhs for Eigen types.
     * 
     * @tparam T Type of the parameters being used in the operation.
     * @param lhs A copy of the first argument is used to return the result.
     * @param rhs A constant reference to the second argument.
     * @return std::enable_if<           is_eigen<T>::value,
     * T>::type The element-wise division of lhs by rhs, if T is of an Eigen type.
     */
    template <typename T>
    typename std::enable_if<           is_eigen<T>::value,
    T>::type               operator_elementwise_division(T lhs, const T &rhs) const { return lhs.array() /= rhs.array(); }

    /**
     * @brief Implements an element-wise division operation of lhs by rhs for vector<arithmetic> types.
     * 
     * @tparam T Type of the elements of the vectors being used in the operation.
     * @param lhs A copy of the first argument is used to return the result.
     * @param rhs A constant reference to the second argument.
     * @return std::enable_if< std::is_arithmetic<T>::value,
     * std::vector<T> >::type The element-wise division of lhs by rhs, if the inputs are of a vector<arithmetic> type.
     */
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    std::vector<T> >::type operator_elementwise_division(std::vector<T> lhs, const std::vector<T> &rhs) const { 
      std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), lhs.begin(), std::divides<T>()); 
      return lhs;
    }


    /**
     * @brief Implements an addition of a scalar of type U to a scalar of type T, returning a type T.
     * 
     * @tparam T Type of the input to which the scalar is added.
     * @tparam U Type of the scalar.
     * @param lhs A copy of the first argument is used to return the result.
     * @param rhs A copy of the scalar, which is added to lhs.
     * @return std::enable_if< std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,
     * T>::type The lhs with the scalar added to it, if the type T is of an arithmetic type and the type U is of 
     * an arithmetic type.
     */
    template <typename T, typename U>
    typename std::enable_if< std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,
    T>::type               operator_scalar_addition(T lhs, U rhs) const { return lhs += rhs; }

    /**
     * @brief Implements an addition of a scalar of type U to every element of an Eigen type, returning an eigen type.
     * 
     * @tparam T Type of the input to which the scalar is added.
     * @tparam U Type of the scalar.
     * @param lhs A copy of the first argument is used to return the result.
     * @param rhs A copy of the scalar, which is added to lhs.
     * @return std::enable_if<           is_eigen<T>::value && std::is_arithmetic<U>::value,
     * T>::type The lhs with the scalar added to its every element, if the type T is of an Eigen type and the type U is of an 
     * arithmetic type.
     */
    template <typename T, typename U>
    typename std::enable_if<           is_eigen<T>::value && std::is_arithmetic<U>::value,
    T>::type               operator_scalar_addition(T lhs, U rhs) const { return lhs.array() += rhs; }

    /**
     * @brief Implements an addition of a scalar of type U to every element of a std::vector<arithmetic> type, returning an 
     * std::vector<arithmetic>
     * 
     * @tparam T Type of the input to which the scalar is added.
     * @tparam U Type of the scalar.
     * @param lhs A copy of the first argument is used to return the result.
     * @param rhs A copy of the scalar, which is added to lhs.
     * @return std::enable_if< std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,
     * std::vector<T> >::type The lhs with the scalar added to its every element, if lhs is of type std::vector<arithmetic> and the
     * type U is of an arithmetic type.
     */
    template <typename T, typename U>
    typename std::enable_if< std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,
    std::vector<T> >::type operator_scalar_addition(std::vector<T> lhs, U rhs) const { 
      for(auto &it : lhs)
        it += rhs;
      return lhs;
    }

    /**
     * @brief Implements a subtraction of a scalar of type U from a scalar of type T, returning a type T.
     * 
     * @tparam T Type of the input from which the scalar is subtracted.
     * @tparam U Type of the scalar.
     * @param lhs A copy of the first argument is used to return the result.
     * @param rhs A copy of the scalar, which is subtracted from lhs.
     * @return std::enable_if< std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,
     * T>::type The lhs with the scalar subtracted from it, if the type T is of an arithmetic type and the type U is of 
     * an arithmetic type.
     */
    template <typename T, typename U>
    typename std::enable_if< std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,
    T>::type               operator_scalar_subtraction(T lhs, U rhs) const { return lhs -= rhs; }

    /**
     * @brief Implements a subtraction of a scalar of type U from every element of an Eigen type, returning an eigen type.
     * 
     * @tparam T Type of the input from which the scalar is subtracted.
     * @tparam U Type of the scalar.
     * @param lhs A copy of the first argument is used to return the result.
     * @param rhs A copy of the scalar, which is subtracted from lhs.
     * @return std::enable_if<           is_eigen<T>::value && std::is_arithmetic<U>::value,
     * T>::type The lhs with the scalar subtracted from its every element, if the type T is of an Eigen type and the type U is of an 
     * arithmetic type.
     */
    template <typename T, typename U>
    typename std::enable_if<           is_eigen<T>::value && std::is_arithmetic<U>::value,
    T>::type               operator_scalar_subtraction(T lhs, U rhs) const { return lhs.array() -= rhs; }

    /**
     * @brief Implements a subtraction of a scalar of type U from every element of a std::vector<arithmetic> type, returning an 
     * std::vector<arithmetic>
     * 
     * @tparam T Type of the input from which the scalar is subtracted.
     * @tparam U Type of the scalar.
     * @param lhs A copy of the first argument is used to return the result.
     * @param rhs A copy of the scalar, which is subtracted from lhs.
     * @return std::enable_if< std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,
     * std::vector<T> >::type The lhs with the scalar subtracted from its every element, if lhs is of type std::vector<arithmetic> and the
     * type U is of an arithmetic type.
     */
    template <typename T, typename U>
    typename std::enable_if< std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,
    std::vector<T> >::type operator_scalar_subtraction(std::vector<T> lhs, U rhs) const { 
      for(auto &it : lhs)
        it -= rhs;
      return lhs;
    }


    /**
     * @brief Implements a division of a scalar of type T by a scalar of type U, returning a type T.
     * 
     * @tparam T Type of the input to be divided by the scalar.
     * @tparam U Type of the scalar.
     * @param lhs A copy of the first argument is used to return the result.
     * @param rhs A copy of the scalar.
     * @return std::enable_if< std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,
     * T>::type The lhs after being divided by the scalar, if the type T is of an arithmetic type and the type U is of 
     * an arithmetic type.
     */
    template <typename T, typename U>
    typename std::enable_if< std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,
    T>::type               operator_scalar_division(T lhs, U rhs) const { return lhs /= rhs; }

    /**
     * @brief Implements a division of every element of an Eigen type by a scalar of type U, returning an Eigen type.
     * 
     * @tparam T Type of the input to be divided by the scalar.
     * @tparam U Type of the scalar.
     * @param lhs A copy of the first argument is used to return the result.
     * @param rhs A copy of the scalar.
     * @return std::enable_if<           is_eigen<T>::value && std::is_arithmetic<U>::value,
     * T>::type The lhs after each of its elements being divided by the scalar, if the type T is of an Eigen type and the 
     * type U is of an arithmetic type.
     */
    template <typename T, typename U>
    typename std::enable_if<           is_eigen<T>::value && std::is_arithmetic<U>::value,
    T>::type               operator_scalar_division(T lhs, U rhs) const { return lhs.array() /= rhs; }

    /**
     * @brief Implements a division of every element of an std::vector<arithmetic> type by a scalar of type U, returning an
     * Eigen type.
     * 
     * @tparam T Type of the input to be divided by the scalar.
     * @tparam U Type of the scalar.
     * @param lhs A copy of the first argument is used to return the result.
     * @param rhs A copy of the scalar.
     * @return std::enable_if< std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,
     * std::vector<T> >::type The lhs after each of its elements being divided by the scalar, if lhs is of an
     * std::vector<arithmetic> type and type U is of an arithmetic type.
     */
    template <typename T, typename U>
    typename std::enable_if< std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,
    std::vector<T> >::type operator_scalar_division(std::vector<T> lhs, U rhs) const { 
      for(auto &it : lhs)
        it /= rhs;
      return lhs;
    }

    /* END BASIC ARITHMETIC OPERATIONS */


    /* BEGIN OTHER ARITHMETIC OPERATIONS */

    /**
     * @brief Implements the square root operation over arg, when it is of an arithmetic type.
     *   
     * @tparam T Type of the input for which the square root is computed.
     * @param arg A constant reference to the input for which the square root is computed.
     * @return std::enable_if< std::is_arithmetic<T>::value,
     * T>::type The square root of the input, if T is of an arithmetic type.
     */
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value, 
    T>::type             square_root(const T &arg) const { return std::sqrt(arg); }

    /**
     * @brief Implements the element-wise square root over each element of arg, when it is of an Eigen type.
     * 
     * @tparam T Type of the input for which the square root is computed.
     * @param arg A constant reference to the input for which the square root is computed. 
     * @return std::enable_if<           is_eigen<T>::value,
     * T>::type An Eigen type where each element is the square root of each respective element of the input arg, if T is
     * of an Eigen type.
     */
    template <typename T>
    typename std::enable_if<           is_eigen<T>::value, 
    T>::type             square_root(const T &arg) const { return Eigen::sqrt(arg.array()); }

    /**
     * @brief Implements the element-wise square root over each element of arg, when it is of type std::vector<arithmetic>.
     * 
     * @tparam T Type of the input for which the square root is computed.
     * @param arg A constant reference to the input for which the square root is computed.  
     * @return std::enable_if< std::is_arithmetic<T>::value,
     * std::vector<T>>::type An std::vector<arithmetic> type where each element is the square root of each respective element 
     * of the input arg, if T is of an arithmetic type and arg is of an std::vector<arithmetic> type.
     */
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value, 
    std::vector<T>>::type square_root(std::vector<T> arg) const { 
      for(auto &it: arg)
        it = std::sqrt(it);
      return arg; 
    }



    /* END OTHER ARITHMETIC OPERATIONS */

    /* BEGIN COMPARISONS */

    /**
     * @brief Implements an operator less than or equal to (<=) for arithmetic types.
     * 
     * @tparam T Type of the parameters being used in the operation.
     * @param lhs A constant reference to the first argument.
     * @param rhs A constant reference to the second argument.
     * @return std::enable_if< std::is_arithmetic<T>::value,
     * bool>::type Returned value is true if lhs <= rhs, it is false otherwise. Assumes the inputs are of an arithmetic type.
     */
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    bool>::type operator_leq(const T &lhs, const T &rhs) const { return lhs <= rhs; }

    /**
     * @brief Implements an operator less than or equal to (<=) for Eigen types. Returns false if any of the elements
     * does not respect the <= condition.
     * 
     * @tparam T Type of the parameters being used in the operation.
     * @param lhs A constant reference to the first argument.
     * @param rhs A constant reference to the second argument
     * @return std::enable_if<           is_eigen<T>::value,
     * bool>::type Returns true if each element of lhs is less than or equal to the corresponding element (in the 
     * same position) of rhs. Returns false otherise. Assumes the inputs are of an Eigen type.
     */
    template <typename T>
    typename std::enable_if<           is_eigen<T>::value,
    bool>::type operator_leq(const T &lhs, const T &rhs) const { return (lhs.array() <= rhs.array()).any(); }

    /**
     * @brief Implements an operator less than or equal to (<=) for std::vector<arithmetic> types. Returns false if
     * any of the elements does not respect the <= condition.
     * 
     * @tparam T Type of the elements in the vectors being used in the operation.
     * @param lhs A constant reference to the first argument.
     * @param rhs A constant reference to the second argument.
     * @return std::enable_if< std::is_arithmetic<T>::value,
     * bool>::type Returns true if each element of lhs is less than or equal to the corresponding element (in the 
     * same position) of rhs. Returns false otherise. Assumes the inputs are of an std::vector<arithmetic> type.
     */
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    bool>::type operator_leq(const std::vector<T> &lhs, const std::vector<T> &rhs) const { 
      for(std::size_t i = 0; i < lhs.size(); ++i)
        if(lhs.at(i) >= rhs.at(i))
          return false;

      return true;
    }

    /**
     * @brief Implements a maximum operation, returning the maximum between two scalars.
     * 
     * @tparam T Type of the parameters being used in the operation.
     * @param lhs A constant reference to the first argument.
     * @param rhs A constant reference to the second argument.
     * @return std::enable_if< std::is_arithmetic<T>::value,
     * T>::type The larger of the two inputs.
     */
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    T>::type               elementwise_max(const T &lhs, const T &rhs) const { return (lhs > rhs) ? lhs : rhs; }

    /**
     * @brief Implements an element-wise maximum operation, returning an Eigen type with the element-wise maximum values 
     * between the inputs.
     * 
     * @tparam T Type of the parameters being used in the operation.
     * @param lhs A constant reference to the first argument.
     * @param rhs A constant reference to the second argument.
     * @return std::enable_if<            is_eigen<T>::value,
     * T>::type An Eigen type, with the element-wise maximum between the two inputs.
     */
    template <typename T>
    typename std::enable_if<            is_eigen<T>::value,
    T>::type               elementwise_max(const T &lhs, const T &rhs) const { return lhs.array().max(rhs.array()); }

    /**
     * @brief Implements an element-wise maximum operation, returning an std::vector<arithmetic> type with the element-wise 
     * maximum values between the inputs.
     * 
     * @tparam T Type of the elements in the vectors being used in the operation.
     * @param lhs A constant reference to the first argument.
     * @param rhs A constant reference to the second argument.
     * @return std::enable_if< std::is_arithmetic<T>::value,
     * std::vector<T> >::type An std::vector<arithmetic> type, with the element-wise maximum between the two inputs.
     */
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    std::vector<T> >::type elementwise_max(std::vector<T> &lhs, const std::vector<T> &rhs) const { 
      std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), lhs.begin(),
        [](const T lhs_element, const T rhs_element){ return (lhs_element > rhs_element) ? lhs_element : rhs_element; } );
      return lhs;
    }

    /**
     * @brief Implements a minimum operation, returning the minimum between two scalars.
     * 
     * @tparam T Type of the parameters being used in the operation.
     * @param lhs A constant reference to the first argument.
     * @param rhs A constant reference to the second argument.
     * @return std::enable_if< std::is_arithmetic<T>::value,
     * T>::type The smaller of the two inputs.
     */
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    T>::type               elementwise_min(const T &lhs, const T &rhs) const { return (lhs < rhs) ? lhs : rhs; }

    /**
     * @brief Implements an element-wise minimum operation, returning an Eigen type with the element-wise minimum values 
     * between the inputs.
     * 
     * @tparam T Type of the parameters being used in the operation.
     * @param lhs A constant reference to the first argument.
     * @param rhs A constant reference to the second argument.
     * @return std::enable_if<            is_eigen<T>::value,
     * T>::type An Eigen type, with the element-wise minimum between the two inputs.
     */
    template <typename T>
    typename std::enable_if<            is_eigen<T>::value,
    T>::type               elementwise_min(const T &lhs, const T &rhs) const { return lhs.array().min(rhs.array()); }

    /**
     * @brief Implements an element-wise minimum operation, returning an std::vector<arithmetic> type with the element-wise 
     * minimum values between the inputs.
     * 
     * @tparam T Type of the elements in the vectors being used in the operation.
     * @param lhs A constant reference to the first argument.
     * @param rhs A constant reference to the second argument.
     * @return std::enable_if< std::is_arithmetic<T>::value,
     * std::vector<T> >::type An std::vector<arithmetic> type, with the element-wise minimum between the two inputs.
     */
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    std::vector<T> >::type elementwise_min(std::vector<T> &lhs, const std::vector<T> &rhs) const { 
      std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), lhs.begin(),
        [](const T lhs_element, const T rhs_element){ return (lhs_element < rhs_element) ? lhs_element : rhs_element; } );
      return lhs;
    }

    /**
     * @brief Checks if the dimensions of the two inputs are the same. Since the inputs are scalars, it always returns true.
     * 
     * @tparam T Type of the parameters being used in the operation.
     * @param lhs A constant reference to the first argument.
     * @param rhs A constant reference to the second argument.
     * @return std::enable_if< std::is_arithmetic<T>::value,
     * bool>::type Always returns true, since the function is only called when the inputs are of an arithmetic type.
     */
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    bool>::type check_dimensions(const T &lhs, const T &rhs) const { return true; }

    /**
     * @brief Checks if the dimensions of the two inputs are the same, for Eigen types.
     * 
     * @tparam T Type of the parameters being used in the operation.
     * @param lhs A constant reference to the first argument.
     * @param rhs A constant reference to the second argument.
     * @return std::enable_if<            is_eigen<T>::value,
     * bool>::type Returns true if both the number of rows and of columns are the same for the two inputs, when the inputs
     * are of an Eigen type. 
     */
    template <typename T>
    typename std::enable_if<            is_eigen<T>::value,
    bool>::type check_dimensions(const T &lhs, const T &rhs) const { return (lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols()); }

    /**
     * @brief Checks if the dimensions of the two inputs are the same, for std::vector<arithmetic> types.
     * 
     * @tparam T Type of the elements in the vectors being used in the operation.
     * @param lhs A constant reference to the first argument.
     * @param rhs A constant reference to the second argument.
     * @return std::enable_if< std::is_arithmetic<T>::value,
     * bool>::type Returns true if both vectors have the same number of elements, when the inputs are of an 
     * std::vector<arithmetic> type.
     */
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    bool>::type check_dimensions(const std::vector<T> &lhs, const std::vector<T> &rhs) const { return (lhs.size() == rhs.size()); }



    /* END COMPARISONS */

    /* BEGIN PRINTING OPERATIONS */

    /**
     * @brief Prints to_print of arithmetic or Eigen type.
     * 
     * @tparam T Type of the input being printed.
     * @param to_print Input to be printed.
     * @return std::enable_if< std::is_arithmetic<T>::value || is_eigen<T>::value,
     * void >::type Returns void.
     */
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value || is_eigen<T>::value,
    void >::type print_datum_t(const T &to_print) const { std::cout << to_print << std::endl; }

    /**
     * @brief Prints to_print of an std::vector<arithmetic> type.
     * 
     * @tparam T Type of the input being printed.
     * @param to_print Input to be printed.
     * @return std::enable_if< std::is_arithmetic<T>::value,
     * void >::type Returns void.
     */
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    void >::type print_datum_t(const std::vector<T> &to_print) const {
      for(const auto &it: to_print)
        std::cout << it << ", ";
      std::cout << "\n" << std::endl;
    }


    /**
     * @brief Prints a to_print vector of arithmetic types
     * 
     * @tparam T Type of the vector elements.
     * @param to_print Input vector to be printed.
     * @return std::enable_if< std::is_arithmetic<T>::value,
     * void >::type Returns void.
     */
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    void >::type print_vector_datum_t(const std::vector<T> &to_print) const {
      for(const auto &it: to_print)
        std::cout << it << ", ";
      std::cout << "\n";
    }

    /**
     * @brief Prints a to_print vector of Eigen types
     * 
     * @tparam T Type of the vector elements.
     * @param to_print Input vector to be printed.
     * @return std::enable_if<           is_eigen<T>::value,
     * void >::type Returns void.
     */
    template <typename T>
    typename std::enable_if<           is_eigen<T>::value,
    void >::type print_vector_datum_t(const std::vector<T> &to_print) const {
      for(const auto &it: to_print)
        std::cout << it << "\n" << std::endl;
    }

    /**
     * @brief Prints a to_print vector of std::vector<arithmetic> types.
     * 
     * @tparam T Type of the elements within each std::vector<arithmetic> being printed.
     * @param to_print Input vector to be printed.
     * @return std::enable_if< std::is_arithmetic<T>::value,
     * void >::type Returns void.
     */
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

    /**
     * @brief Gets the dimensions from a scalar datum. Always returns 1 since this function is only called for scalars.
     * 
     * @tparam T Input type
     * @param datum Constant reference to the argument.
     * @return std::enable_if< std::is_arithmetic<T>::value,
     * int>::type Number of dimensions. Always 1.
     */
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    int>::type get_dimensions(const T &datum) const
    { return 1; }

    /**
     * @brief Gets the dimensions from an Eigen input, that is, the product between the rows and columns of datum.
     * 
     * @tparam T Input type.
     * @param datum Constant reference to the argument.
     * @return std::enable_if<            is_eigen<T>::value,
     * int>::type Number of dimensions. Number of rows times number of columns of datum.
     */
    template <typename T>
    typename std::enable_if<            is_eigen<T>::value,
    int>::type get_dimensions(const T &datum) const
    { return datum.rows() * datum.cols(); }

    /**
     * @brief Get the dimensions from an std::vector<arithmetic> input, that is, the number of elements in the vector.
     * 
     * @tparam T Type of the vector elements.
     * @param datum Constant reference to the argument.
     * @return std::enable_if< std::is_arithmetic<T>::value,
     * int>::type Number of dimensions. Number of elements in the vector.
     */
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    int>::type get_dimensions(const std::vector<T> &datum) const
    { return datum.size(); }

    /* END OTHER OPERATIONS */
};

} // namespace tudat_learn

#endif // TUDAT_LEARN_OPERATOR_HPP