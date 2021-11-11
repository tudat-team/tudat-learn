/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_TYPES_H
#define TUDAT_LEARN_TYPES_H

#include <vector>


namespace tudat_learn
{
  /// Alias for an std::vector of type double
  using vector_double = std::vector<double>;

  /// Alias for an std::vector of vector_double, vector_double std::pairs
  using data_t = std::vector< std::pair< std::vector< double >, std::vector< double > > >;

} // TUDAT_LEARN_TYPES_H

#endif // TUDAT_LEARN_TYPES_H