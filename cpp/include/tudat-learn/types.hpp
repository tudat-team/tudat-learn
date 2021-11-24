/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_TYPES_HPP
#define TUDAT_LEARN_TYPES_HPP

#include <vector>
#include <variant>


namespace tudat_learn
{
  /// Alias for an std::vector of type double
  using vector_double = std::vector<double>;

  // Alias for an std::vector of type int
  using vector_int = std::vector<int>;

  // None type to be used in unlabelled datasets
  typedef struct None_t { } none_t;



} // namespace tudat_learn

#endif // TUDAT_LEARN_TYPES_HPP