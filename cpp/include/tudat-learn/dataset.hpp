/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_DATASET_HPP
#define TUDAT_LEARN_DATASET_HPP

#include <map>
#include <vector>

#include "tudat-learn/types.hpp"

namespace tudat_learn
{
  
class Dataset {
  
  dataset_t data;

  public:
    Dataset(const dataset_t &dataset);

    /**
     * @brief Access datum_t at position n in the vector of dataset_t.
     * 
     * @param n 
     * @return datum_t& reference to the datum_t element
     */
    datum_t &at(const size_t n) {
      return data.at(n);
    }

    /**
     * @brief Access datum_t at position n in the vector of dataset_t.
     * 
     * @param n 
     * @return const datum_t& constant reference to the datum_t element
     */
    const datum_t &at(const size_t n) const {
      return data.at(n);
    }
};
  
} // namespace tudat_learn


#endif // TUDAT_LEARN_DATASET_HPP
