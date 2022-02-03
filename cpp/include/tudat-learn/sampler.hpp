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


#include <iterator>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "tudat-learn/operator.hpp"
#include "tudat-learn/random.hpp"
#include "tudat-learn/types.hpp"

namespace tudat_learn
{
  
template <typename Datum_t>
class Sampler : public Operator<Datum_t> {
  public:
    Sampler(const std::pair<Datum_t, Datum_t> &range) 
    : Operator<Datum_t>() {
      set_range(range);
    }

    virtual std::vector<Datum_t> sample( ) const = 0;

  protected:
    void set_range(const std::pair<Datum_t, Datum_t> &range) {
      if(this->operator_leq(range.second, range.first)) throw std::runtime_error("LatinHypercubeSampler range must have the (min, max) form, with every element of min being smaller than the corresponding element of max, for multidimensional types.");
      
      this->range = range;
      range_size = this->operator_difference(range.second, range.first);
    }

    // randomly generates a Datum_t between 0 and 1 for arithmetic types
    template <typename Datum_tt = Datum_t>
    typename std::enable_if< std::is_arithmetic<Datum_tt>::value,
    Datum_tt >::type sample_zero_one( ) const {
      std::uniform_real_distribution<typename Datum_tt::Scalar> uniform(0,1);
      return uniform(Random::get_rng());
    }

    // randomly generates a Datum_t of eigen type with all elements between 0 and 1
    template <typename Datum_tt = Datum_t>
    typename std::enable_if<           is_eigen<Datum_tt>::value,
    Datum_tt >::type sample_zero_one( ) const {
      std::uniform_real_distribution<typename Datum_tt::Scalar> uniform(0,1);
      return Datum_tt::NullaryExpr(range.first.rows(), range.first.cols(),
              [&](){ return uniform(Random::get_rng()); });
    }

    // randomly generates a Datum_t of vector<arithmetic> type with all elements between 0 and 1
    template <typename Datum_tt = Datum_t>
    typename std::enable_if<      is_stl_vector<Datum_tt>::value,
    Datum_tt >::type sample_zero_one( ) const {
      Datum_tt sample;
      sample.reserve(range.first.size());

      std::uniform_real_distribution<typename Datum_tt::value_type> uniform(0,1);
      std::generate_n(std::back_inserter(sample), range.first.size(), 
        [&](){ return uniform(Random::get_rng()); });

      return sample;
    }

  protected:
    std::pair<Datum_t, Datum_t> range;

    Datum_t range_size;
};
  
} // namespace tudat_learn

#endif // TUDAT_LEARN_SAMPLER_HPP
