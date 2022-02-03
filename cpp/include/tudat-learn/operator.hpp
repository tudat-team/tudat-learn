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

#include <type_traits>
#include <vector>

#include <Eigen/Core>

#include "tudat-learn/random.hpp"
#include "tudat-learn/types.hpp"

namespace tudat_learn
{

template <typename Datum_t>
class Operator {
  protected:
    Operator() { }

    /* BEGIN ARITHMETIC OPERATIONS */

    // implemets an operator difference for arithmetic and eigen types
    template <typename Datum_tt = Datum_t>
    typename std::enable_if< std::is_arithmetic<Datum_tt>::value || is_eigen<Datum_tt>::value,
    Datum_t >::type operator_difference(const Datum_tt &lhs, const Datum_tt &rhs) const { return lhs - rhs; }

    // implements an operator difference for vector<arithmetic> types
    template <typename Datum_tt = Datum_t>
    typename std::enable_if<      is_stl_vector<Datum_tt>::value && std::is_arithmetic<typename Datum_tt::value_type>::value,
    Datum_t >::type operator_difference(Datum_tt lhs, const Datum_tt &rhs) const {
      for(int i = 0; i < lhs.size(); ++i)
        lhs.at(i) -= rhs.at(i);
      return lhs;
    }

    // implements an operator sum for arithmetic and eigen types
    template <typename Datum_tt = Datum_t>
    typename std::enable_if< std::is_arithmetic<Datum_tt>::value || is_eigen<Datum_tt>::value,
    Datum_t >::type operator_addition(const Datum_tt &lhs, const Datum_tt &rhs) const { return lhs + rhs; }

    // implements an operator sum for vector<arithmetic> types
    template <typename Datum_tt = Datum_t>
    typename std::enable_if<      is_stl_vector<Datum_tt>::value && std::is_arithmetic<typename Datum_tt::value_type>::value,
    Datum_t >::type operator_addition(Datum_tt lhs, const Datum_tt &rhs) const {
      for(int i = 0; i < lhs.size(); ++i)
        lhs.at(i) += rhs.at(i);
      return lhs;
    }

    // implements a multiplication for arithmetic types
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    T>::type operator_elementwise_multiplication(const T &lhs, const T &rhs) const { return lhs * rhs; }

    // implements an element-wise multiplication for eigen types
    template <typename T>
    typename std::enable_if<           is_eigen<T>::value,
    T>::type operator_elementwise_multiplication(const T &lhs, const T &rhs) const { return lhs.array() * rhs.array(); }

    // implements an element-wise multiplication for vector<arithmetic> types
    template <typename Datum_tt = Datum_t>
    typename std::enable_if<      is_stl_vector<Datum_tt>::value,
    Datum_tt>::type operator_elementwise_multiplication(Datum_tt lhs, const Datum_tt &rhs) const { 
      std::transform(lhs.cbegin(), lhs.cend(), rhs.cbegin(), lhs.begin(), std::multiplies<typename Datum_tt::value_type>()); 
      return lhs;
    }

    // implements a division for arithmetic types
    template <typename T, typename U>
    typename std::enable_if< std::is_arithmetic<T>::value && std::is_arithmetic<U>::value,
    T>::type operator_scalar_division(const T &lhs, const U &rhs) const { return lhs / rhs; }

    // implements an element-wise division by a scalar for eigen types
    template <typename T, typename U>
    typename std::enable_if<           is_eigen<T>::value && std::is_arithmetic<U>::value,
    T>::type operator_scalar_division(const T &lhs, const U &rhs) const { return lhs.array() / rhs; }

    // implements an element-wise division by a scalar for vector<arithmetic> types
    template <typename Datum_tt = Datum_t, typename U>
    typename std::enable_if<      is_stl_vector<Datum_tt>::value  && std::is_arithmetic<U>::value,
    Datum_tt>::type operator_scalar_division(Datum_tt lhs, const U &rhs) const { 
      for(auto &it : lhs)
        it /= rhs;
      return lhs;
    }

    /* END ARITHMETIC OPERATIONS */

    /* BEGIN COMPARISONS */
    /* END COMPARISONS */
};

} // namespace tudat_learn

#endif // TUDAT_LEARN_OPERATOR_HPP