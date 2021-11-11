/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_DATASET_H
#define TUDAT_LEARN_DATASET_H

#include <map>
#include <vector>

#include "types.h"

namespace tudat_learn
{
  
class Dataset {
  
  data_t data;

  public:
    Dataset();

    Dataset(const data_t &dataset);
};
  
} // namespace tudat_learn


#endif // TUDAT_LEARN_DATASET_H
