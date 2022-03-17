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
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "tudat-learn/random.hpp"
#include "tudat-learn/sampler.hpp"
#include "tudat-learn/types.hpp"

namespace tudat_learn
{

/**
 * @brief LatinHypercubeSampler class. \n 
 * Implements Latin Hypercube Sampling for an arbitrary amount of dimensions. https://en.wikipedia.org/wiki/Latin_hypercube_sampling
 * 
 * @tparam Datum_t Type of the data/feature vectors being sampled.
 */
template <typename Datum_t>
class LatinHypercubeSampler : public Sampler<Datum_t> {
  public:

    /**
     * @brief Deleted default constructor, to ensure the LatinHypercubeSampler is created with settings.
     * 
     */
    LatinHypercubeSampler() = delete;

    /**
     * @brief Constructor that sets the range from which feature vectors are sampled, and how many are supposed to be sampled.
     * The first element of the range must be feature-wise lower than its second element.
     * 
     * @tparam Datum_tt Same as Datum_t.
     * @tparam std::enable_if_t<
     * is_floating_point_eigen<Datum_tt>()       ||
     * std::is_floating_point<Datum_tt>::value   ||
     * is_floating_point_stl_vector<Datum_tt>()
     * > The constructor is enabled if and only if Datum_tt is of a floating-point scalar, floating-point Eigen, or 
     * floating-point std::vector type.
     * @param range Range from which feature-vectors will be sampled.
     * @param number_samples How many samples will be generated.
     */
    template <
      typename Datum_tt = Datum_t,
      typename = std::enable_if_t< 
        is_floating_point_eigen<Datum_tt>()       || 
        std::is_floating_point<Datum_tt>::value   ||
        is_floating_point_stl_vector<Datum_tt>()
      >
    >
    LatinHypercubeSampler(
      const std::pair<Datum_tt, Datum_tt> &range,
      const int number_samples
    ) :
    Sampler<Datum_tt>(range) {
      if(!this->check_dimensions(range.first, range.second)) throw std::runtime_error("The lower and upper bounds of the range in the LatinHypercubeSampler must have the same dimensions.");

      set_buckets(number_samples);
    }


    /**
     * @brief Sample number_samples feature vectors of Datum_t types, according to Latin Hypercube Sampling.
     * https://en.wikipedia.org/wiki/Latin_hypercube_sampling
     * 
     * @return std::vector<Datum_t> Vector with the sampled feature vectors.
     */
    virtual std::vector<Datum_t> sample( ) const override;

    /**
     * @brief Sample method that accepts a range and a number of samples different from the ones provided in the constructor.
     * Saves the new range and new number of samples. Calls sample( ).
     * 
     * @param new_range Constant reference to the new range.
     * @param number_samples Constant copy of the desired number of samples.
     * @return std::vector<Datum_t> Vector with the sampled feature vectors.
     */
    virtual std::vector<Datum_t> sample(const std::pair<Datum_t, Datum_t> &new_range, const int number_samples);

    /**
     * @brief Sample method that accepts a range different from the one provided in the constructor.
     * Saves the new range. Calls sample( ).
     * 
     * @param new_range Constant reference to the new range.
     * @return std::vector<Datum_t> Vector with the sampled feature vectors.
     */
    virtual std::vector<Datum_t> sample(const std::pair<Datum_t, Datum_t> &new_range                          );

   /**
     * @brief Sample method that accepts a number of samples different from the one provided in the constructor.
     * Saves the new number of samples. Calls sample( ).
     * 
     * @param number_samples Constant copy of the desired number of samples.
     * @return std::vector<Datum_t> Vector with the sampled feature vectors.
     */
    virtual std::vector<Datum_t> sample(                                              const int number_samples);

  protected:

    /**
     * @brief Sets the buckets, making sure there is one or more bucket per dimension.
     * 
     * @param buckets_per_dimension Number of buckets per dimension, equivalent to the number of samples.
     */
    void set_buckets(const int buckets_per_dimension) {
      if(buckets_per_dimension < 1) throw std::runtime_error("LatinHypercubeSampler must have one or more buckets per dimension.");
      
      this->buckets_per_dimension = buckets_per_dimension;
      bucket_size = this->operator_scalar_division(this->range_size, buckets_per_dimension);
    }

    
    /**
     * @brief Implements a function that generates a vector of arithmetic types. The vector contains a number of elements equal
     * to the number of desired samples. Each element contains, for each feature, the bucket from which it is going to be 
     * sampled. \n 
     * Receives as an input, a vector of vectors of integers. Each vector of integers corresponds to a different feature. Each
     * of the integers in each of those vectors corresponds to a bucket index, containing the order in which they are sampled.
     * Could definitely use a diagram.
     * 
     * @tparam Datum_tt Same as Datum_t.
     * @param sampled_indices 
     * @return std::enable_if< std::is_arithmetic<Datum_tt>::value,
     * std::vector<Datum_tt> >::type 
     */
    template <typename Datum_tt = Datum_t>
    typename std::enable_if< std::is_arithmetic<Datum_tt>::value,
    std::vector<Datum_tt> >::type generate_buckets(const std::vector<std::vector<size_t>> &sampled_indices) const;

    /**
     * @brief Implements a function that generates a vector of Eigen types. The vector contains a number of elements equal
     * to the number of desired samples. Each element contains, for each feature, the bucket from which it is going to be 
     * sampled. \n 
     * Receives as an input, a vector of vectors of integers. Each vector of integers corresponds to a different feature. Each
     * of the integers in each of those vectors corresponds to a bucket index, containing the order in which they are sampled.
     * Could definitely use a diagram.
     * 
     * @tparam Datum_tt Same as Datum_t.
     * @param sampled_indices 
     * @return std::enable_if< std::is_arithmetic<Datum_tt>::value,
     * std::vector<Datum_tt> >::type 
     */
    template <typename Datum_tt = Datum_t>
    typename std::enable_if<           is_eigen<Datum_tt>::value,
    std::vector<Datum_tt> >::type generate_buckets(const std::vector<std::vector<size_t>> &sampled_indices) const;

    /**
     * @brief Implements a function that generates a vector of vector<arithmetic> types. The vector contains a number of elements equal
     * to the number of desired samples. Each element contains, for each feature, the bucket from which it is going to be 
     * sampled. \n 
     * Receives as an input, a vector of vectors of integers. Each vector of integers corresponds to a different feature. Each
     * of the integers in each of those vectors corresponds to a bucket index, containing the order in which they are sampled.
     * Could definitely use a diagram.
     * 
     * @tparam Datum_tt Same as Datum_t.
     * @param sampled_indices 
     * @return std::enable_if< std::is_arithmetic<Datum_tt>::value,
     * std::vector<Datum_tt> >::type 
     */
    template <typename Datum_tt = Datum_t>
    typename std::enable_if<      is_stl_vector<Datum_tt>::value && std::is_arithmetic<typename Datum_tt::value_type>::value,
    std::vector<Datum_tt> >::type generate_buckets(const std::vector<std::vector<size_t>> &sampled_indices) const;


  protected:
    int buckets_per_dimension; /**< Number of buckets per dimension, equivalent to the number of samples. */

    Datum_t bucket_size;       /** Feature-wise bucket size. */


};

} // namespace tudat_learn

#include "tudat-learn/samplers/random_samplers/latin_hypercube_sampler.tpp"

#endif // TUDAT_LEARN_LATIN_HYPERCUBE_SAMPLER_HPP