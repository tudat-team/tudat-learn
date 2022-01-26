/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_SAMPLING_HPP
#define TUDAT_LEARN_SAMPLING_HPP

#include <type_traits>
#include <vector>

#include "tudat-learn/types.hpp"

namespace tudat_learn
{
  
template <typename Datum_t>
class Sampler {
  public:
    virtual std::vector<Datum_t> sample( ) const = 0;

  protected:

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
        if(lhs.at(i) <= rhs.at(i))
          return false;

      return true;
    }
};
  
} // namespace tudat_learn

#endif // TUDAT_LEARN_SAMPLING_HPP
