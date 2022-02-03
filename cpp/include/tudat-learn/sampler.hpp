/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_SAMPLER_HPP
#define TUDAT_LEARN_SAMPLER_HPP

#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "tudat-learn/random.hpp"
#include "tudat-learn/types.hpp"

namespace tudat_learn
{
  
template <typename Datum_t>
class Sampler {
  public:
    Sampler(const std::pair<Datum_t, Datum_t> &range) {
      set_range(range);
    }

    virtual std::vector<Datum_t> sample( ) const = 0;

  protected:
    void set_range(const std::pair<Datum_t, Datum_t> &range) {
      if(this->operator_leq(range.second, range.first)) throw std::runtime_error("LatinHypercubeSampler range must have the (min, max) form, with every element of min being smaller than the corresponding element of max, for multidimensional types.");
      
      this->range = range;
      range_size = this->operator_difference(range.second, range.first);
    }

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

    // implements a method that retrieves the dimension if the Datum_t is of arithmetic types
    template <typename Datum_tt=Datum_t>
    typename std::enable_if< std::is_arithmetic<Datum_tt>::value,
    int>::type get_dimensions(const std::pair<     Datum_tt,        Datum_tt> &range) const
    { return 1; }

    // implements a method that retrieves the dimension if the Datum_t is of eigen types
    template <typename Datum_tt=Datum_t>
    typename std::enable_if<            is_eigen<Datum_tt>::value,
    int>::type get_dimensions(const std::pair<     Datum_tt,        Datum_tt> &range) const
    { return range.first.rows() * range.first.cols(); }

    // implements a method that retrieves the dimension if the Datum_t is of arithmetic types
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    int>::type get_dimensions(const std::pair<std::vector<T>, std::vector<T>> &range) const
    { return range.first.size(); }

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
    Datum_t >::type operator_sum(const Datum_tt &lhs, const Datum_tt &rhs) const { return lhs + rhs; }

    // implements an operator sum for vector<arithmetic> types
    template <typename Datum_tt = Datum_t>
    typename std::enable_if<      is_stl_vector<Datum_tt>::value && std::is_arithmetic<typename Datum_tt::value_type>::value,
    Datum_t >::type operator_sum(Datum_tt lhs, const Datum_tt &rhs) const {
      for(int i = 0; i < lhs.size(); ++i)
        lhs.at(i) += rhs.at(i);
      return lhs;
    }

    // generates a Datum_t between 0 and 1 for arithmetic types
    template <typename Datum_tt = Datum_t>
    typename std::enable_if< std::is_arithmetic<Datum_tt>::value,
    Datum_tt >::type sample_zero_one( ) const {
      std::uniform_real_distribution<typename Datum_tt::Scalar> uniform(0,1);
      return uniform(Random::get_rng());
    }

    template <typename Datum_tt = Datum_t>
    typename std::enable_if<           is_eigen<Datum_tt>::value,
    Datum_tt >::type sample_zero_one( ) const {
      std::uniform_real_distribution<typename Datum_tt::Scalar> uniform(0,1);
      return Datum_tt::NullaryExpr(range.first.rows(), range.first.cols(),
              [&](){ return uniform(Random::get_rng()); });
    }

    template <typename Datum_tt = Datum_t>
    typename std::enable_if<      is_stl_vector<Datum_tt>::value,
    Datum_tt >::type sample_zero_one( ) const {
    
    }

  protected:
    std::pair<Datum_t, Datum_t> range;

    Datum_t range_size;
};
  
} // namespace tudat_learn

#endif // TUDAT_LEARN_SAMPLER_HPP
