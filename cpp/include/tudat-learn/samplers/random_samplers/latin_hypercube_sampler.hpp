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

#include <algorithm>
#include <numeric>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "tudat-learn/random.hpp"
#include "tudat-learn/sampler.hpp"
#include "tudat-learn/types.hpp"

namespace tudat_learn
{

template <typename Datum_t>
class LatinHypercubeSampler : public Sampler<Datum_t> {
  public:
    LatinHypercubeSampler() = delete;

    template <
      typename Datum_tt = Datum_t,
      typename = std::enable_if_t< 
        (is_eigen<Datum_tt>::value &&  std::is_floating_point<typename Datum_tt::value_type>::value) || 
                                       std::is_floating_point<         Datum_tt           >::value   ||
        (is_stl_vector<Datum_tt>::value && std::is_arithmetic<typename Datum_tt::value_type>::value) 
      >
    >
    LatinHypercubeSampler(
      const std::pair<Datum_tt, Datum_tt> &range,
      const int number_samples
    ) :
    Sampler<Datum_tt>(range) {
      set_buckets(number_samples);
    }


    virtual std::vector<Datum_t> sample( ) const override;

    virtual std::vector<Datum_t> sample(const std::pair<Datum_t, Datum_t> &new_range, const int number_samples);

    int test() const { return this->get_dimensions(this->range); }

  protected:
    void set_buckets(const int buckets_per_dimension) {
      if(buckets_per_dimension < 1) throw std::runtime_error("LatinHypercubeSampler must have one or more buckets per dimension.");
      
      this->buckets_per_dimension = buckets_per_dimension;
      bucket_size = this->operator_scalar_division(this->range_size, buckets_per_dimension);
    }

    
    // generates a vector of Datum_t with the selected bucket indices for arithmetic types
    template <typename Datum_tt = Datum_t>
    typename std::enable_if< std::is_arithmetic<Datum_tt>::value,
    std::vector<Datum_tt> >::type generate_buckets(const std::vector<std::vector<int>> &sampled_indices) const;

    // generates a vector of Datum_t with the selected bucket indices for for eigen types
    template <typename Datum_tt = Datum_t>
    typename std::enable_if<           is_eigen<Datum_tt>::value,
    std::vector<Datum_tt> >::type generate_buckets(const std::vector<std::vector<int>> &sampled_indices) const;

    // generates a vector of Datum_t with the selected bucket indices for vector<arithmetic> types
    template <typename Datum_tt = Datum_t>
    typename std::enable_if<      is_stl_vector<Datum_tt>::value && std::is_arithmetic<typename Datum_tt::value_type>::value,
    std::vector<Datum_tt> >::type generate_buckets(const std::vector<std::vector<int>> &sampled_indices) const;

    // print a vector of arithmetic types
    template <typename Datum_tt = Datum_t>
    typename std::enable_if< std::is_arithmetic<Datum_tt>::value,
    void >::type print_vector_datum_t(const std::vector<Datum_tt> &to_print) const {
      for(const auto &it: to_print)
        std::cout << it << ", ";

      std::cout << "\n";
    }

    // print a vector of eigen types
    template <typename Datum_tt = Datum_t>
    typename std::enable_if<           is_eigen<Datum_tt>::value,
    void >::type print_vector_datum_t(const std::vector<Datum_tt> &to_print) const {
      for(const auto &it: to_print)
        std::cout << it << "\n" << std::endl;
    }

    // print a vector of vector<arithmetic> types
    template <typename Datum_tt = Datum_t>
    typename std::enable_if<      is_stl_vector<Datum_tt>::value && std::is_arithmetic<typename Datum_tt::value_type>::value,
    void >::type print_vector_datum_t(const std::vector<Datum_tt> &to_print) const {
      for(const auto &it: to_print) {
        for(const auto &itt: it)
          std::cout << itt << ", ";

        std::cout << "\n" << std::endl;
      }
    }

    // print an arithmetic type
    template <typename Datum_tt = Datum_t>
    typename std::enable_if< std::is_arithmetic<Datum_tt>::value,
    void >::type print_datum_t(const Datum_tt &to_print) const {
      std::cout << to_print << std::endl;
    }

    // print an eigen type
    template <typename Datum_tt = Datum_t>
    typename std::enable_if<           is_eigen<Datum_tt>::value,
    void >::type print_datum_t(const Datum_tt &to_print) const {
      std::cout << to_print << std::endl;
    }

    // print a vector<arithmetic> type
    template <typename Datum_tt = Datum_t>
    typename std::enable_if<      is_stl_vector<Datum_tt>::value && std::is_arithmetic<typename Datum_tt::value_type>::value,
    void >::type print_datum_t(const Datum_tt &to_print) const {
      for(const auto &it: to_print)
        std::cout << it << ", ";
      std::cout << "\n" << std::endl;
    }
    

  protected:
    int buckets_per_dimension;

    Datum_t bucket_size;


};

} // namespace tudat_learn

#include "tudat-learn/samplers/random_samplers/latin_hypercube_sampler.tpp"

#endif // TUDAT_LEARN_LATIN_HYPERCUBE_SAMPLER_HPP