/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_MINMAX_SCALER_HPP
#define TUDAT_LEARN_MINMAX_SCALER_HPP

#include <stdexcept>

#include "tudat-learn/processing/scaler.hpp"

namespace tudat_learn
{

template <typename Datum_t, typename Label_t>
class MinmaxScaler : Scaler<Datum_t, Label_t> {
  public:
    MinmaxScaler( ) :
    range(std::pair(0, 1)) 
    { }

    MinmaxScaler( 
      const std::pair<int, int> &range // maybe floating-point?
    ) :
    range(range) {
      if(range.first >= range.second) throw std::runtime_error("Minmax range must have the (min, max) form, with min < max. Please choose a valid range.");
    }


  private:
    std::pair<int, int> range;

};

} // namespace tudat_learn

#endif // TUDAT_LEARN_MINMAX_SCALER_HPP