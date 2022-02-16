/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */


#ifndef TUDAT_LEARN_RANDOM_HPP
#define TUDAT_LEARN_RANDOM_HPP

#include <random>

namespace tudat_learn 
{

/**
 * @brief Random class. \n
 * Is supposed to be used in every class within tudat-learn which uses RNGs. Only has static methods, needs not to be instatiated.
 * 
 */
struct Random {

  /**
   * @brief Sets the new seed for every tudat-learn appliation. Also sets rng with the new seed.
   * 
   * @param new_seed New seed.
   */
  static void set_seed(unsigned int new_seed) {
    seed = new_seed;
    rng = std::mt19937(seed);
  }
  
  /**
   * @brief Returns a copy of the seed.
   * 
   * @return unsigned int Seed.
   */
  static unsigned int  get_seed( ) { return seed; }

  /**
   * @brief Returns a reference to the rng, which is used to randomly generate numbers in the code.
   * 
   * @return std::mt19937& Reference to rng.
   */
  static std::mt19937 &get_rng ( ) { return rng; }

  private:
    static unsigned int seed; /**< Random seed, initialized to 0 by default, in random.cpp. */
    static std::mt19937 rng;  /**< Mersenne-Twister RNG, initialized with the 0 seed by default, in random.cpp. */
};

} // namespace tudat_learn

#endif // TUDAT_LEARN_RANDOM_HPP