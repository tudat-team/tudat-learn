// /*    Copyright (c) 2010-2018, Delft University of Technology
//  *    All rights reserved
//  *
//  *    This file is part of the Tudat. Redistribution and use in source and
//  *    binary forms, with or without modification, are permitted exclusively
//  *    under the terms of the Modified BSD license. You should have received
//  *    a copy of the license with this file. If not, please or visit:
//  *    http://tudat.tudelft.nl/LICENSE.
//  */

// #ifndef TUDAT_LEARN_RANDOM_SAMPLER_HPP
// #define TUDAT_LEARN_RANDOM_SAMPLER_HPP

// #include <random>
// #include <type_traits>
// #include <utility>
// #include <vector>

// #include "tudat-learn/random.hpp"
// #include "tudat-learn/sampler.hpp"
// #include "tudat-learn/types.hpp"

// namespace tudat_learn
// {

// template <typename Datum_t>
// class RandomSampler : public Sampler<Datum_t> {
//   public:
//     RandomSampler(
//       const std::pair<Datum_t, Datum_t> &range,
//       const unsigned int seed = Random::get_seed()
//     ) : Sampler<Datum_t>(range) {
//       set_seed(seed);
//     }

//   protected:
    

//   protected:
//     unsigned int seed;

//     mutable std::mt19937 rng; // has to be mutable due to const-correctness on the Sampler<Datum_t>::sample() method.
// };

// } // namespace tudat_learn

// #endif // TUDAT_LEARN_RANDOM_SAMPLER_HPP