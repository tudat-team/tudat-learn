/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_LATIN_HYPERCUBE_SAMPLER_HPP
#define TUDAT_LEARN_LATIN_HYPERCUBE_SAMPLER_HPP

#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "tudat-learn/samplers/random_sampler.hpp"
#include "tudat-learn/types.hpp"

namespace tudat_learn
{

template <typename Datum_t>
class LatinHypercubeSampler : public RandomSampler<Datum_t> {
  public:
    LatinHypercubeSampler() = delete;

    LatinHypercubeSampler(
      const std::pair<Datum_t, Datum_t> &range,
      const int number_samples,
      const unsigned int seed = Random::seed
    ) :
    RandomSampler<Datum_t>(range, seed) {
      set_buckets(number_samples);
    }


    virtual std::vector<Datum_t> sample( ) const override;

    virtual std::vector<Datum_t> sample(const std::pair<Datum_t, Datum_t> &new_range, const int number_samples);

    int test() const { return get_dimensions(this->range); }

  private:
    void set_buckets(const int buckets_per_dimension) {
      if(buckets_per_dimension < 1) throw std::runtime_error("LatinHypercubeSampler must have one or more buckets per dimension.");
      
      this->buckets_per_dimension = buckets_per_dimension;
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

    

  protected:
    int buckets_per_dimension;

    Datum_t bucket_size;


};

} // namespace tudat_learn

#include "tudat-learn/samplers/random_samplers/latin_hypercube_sampler.tpp"

#endif // TUDAT_LEARN_LATIN_HYPERCUBE_SAMPLER_HPP