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

struct Random {
  static void set_seed(unsigned int new_seed) {
    seed = new_seed;
    rng = std::mt19937(seed);
  }
  
  static int get_seed( ) { return seed; }
  static std::mt19937 &get_rng( ) { return rng; }

  private:
    static unsigned int seed;
    static std::mt19937 rng;
};


} // namespace tudat_learn

#endif // TUDAT_LEARN_RANDOM_HPP