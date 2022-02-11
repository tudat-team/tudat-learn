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

/**
 * @brief Base Sampler class. \n
 * Provides a base class implementation for all the samplers in tudat-learn. Inherits from Operator as it needs the operations
 * to be uniform across different Datum_t types. Inherits from Operator.
 * 
 * @tparam Datum_t The type of a single feature vector. Can be a simple scalar, an std::vector<T>, an Eigen::Matrix, ...
 */
template <typename Datum_t>
class Sampler : public Operator<Datum_t> {
  public:

    /**
     * @brief Deleted default constructor, to force a range selection.
     * 
     */
    Sampler( ) = delete;

    /**
     * @brief Constructor with the range from which the elements are sampled.
     * 
     * @param range Constant reference to a pair of Datum_t types, where the first one must be element-wise smaller than the second one.
     */
    Sampler(const std::pair<Datum_t, Datum_t> &range) 
    : Operator<Datum_t>() {
      set_range(range);
    }

    /**
     * @brief Virtual destructor, as the class has virtual methods.
     * 
     */
    virtual ~Sampler( ) { }

    /**
     * @brief Pure virtual method to sample feature vectors of Datum_t types. The amount, along with the definitions, are expected to be
     * given by the constructors of the classes that are derivded from Sampler.
     * 
     * @return std::vector<Datum_t> Vector with the sampled feature vectors.
     */
    virtual std::vector<Datum_t> sample( ) const = 0;

  protected:
    /**
     * @brief Sets the range from which the elements are sampled. \n
     * Can be used by the derived classes. Verifies if each new element of range.first is smaller than the corresponding
     * element of range.second. Computes the range_size, that is, the difference between range.second and range.first.
     * 
     * @param range Constant reference to the new range.
     */
    void set_range(const std::pair<Datum_t, Datum_t> &range) {
      if(this->operator_leq(range.second, range.first)) throw std::runtime_error("LatinHypercubeSampler range must have the (min, max) form, with every element of min being smaller than the corresponding element of max, for multidimensional types.");
      
      this->range = range;
      range_size = this->operator_difference(range.second, range.first);
    }

    /**
     * @brief Generates a random number of an arithmetic type between 0 and 1, drawn from an uniform distribution. 
     * Was not necessarily tested for integer types.
     * 
     * @tparam Datum_tt Type of the value being sampled.
     * @return std::enable_if< std::is_arithmetic<Datum_tt>::value,
     * Datum_tt >::type Random arithmetic type between 0 and 1.
     */
    template <typename Datum_tt = Datum_t>
    typename std::enable_if< std::is_arithmetic<Datum_tt>::value,
    Datum_tt >::type sample_zero_one( ) const {
      std::uniform_real_distribution<Datum_tt> uniform(0,1);
      return uniform(Random::get_rng());
    }


    /**
     * @brief Generates an Eigen structure with each element being randomly sampled between 0 and 1, drawn from an uniform
     * distribution. Was not necessarily tested for integer types.
     * 
     * @tparam Datum_tt Type of the value being sampled.
     * @return std::enable_if<           is_eigen<Datum_tt>::value,
     * Datum_tt >::type Eigen type with each element between 0 and 1.
     */
    template <typename Datum_tt = Datum_t>
    typename std::enable_if<           is_eigen<Datum_tt>::value,
    Datum_tt >::type sample_zero_one( ) const {
      std::uniform_real_distribution<typename Datum_tt::Scalar> uniform(0,1);
      return Datum_tt::NullaryExpr(range.first.rows(), range.first.cols(),
              [&](){ return uniform(Random::get_rng()); });
    }

    /**
     * @brief Generates an std::vector<arithmetic> with each element being randomly sampled between 0 and 1, drawn from an
     * uniform distribution. Was not necessarily tested for integer types. 
     * 
     * @tparam Datum_tt Type of the value being sampled.
     * @return std::enable_if<      is_stl_vector<Datum_tt>::value,
     * Datum_tt >::type std::vector<arithmetic> type with each element between 0 and 1.
     */
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
    std::pair<Datum_t, Datum_t> range; /**< Range from which the elements are sampled. */

    Datum_t range_size;                /**< Range size, difference between range.second and range.first. */
};
  
} // namespace tudat_learn

#endif // TUDAT_LEARN_SAMPLER_HPP
