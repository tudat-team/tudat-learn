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
#include <utility>
#include <vector>



#include "tudat-learn/processing/scaler.hpp"

namespace tudat_learn
{

template <typename Datum_t, typename Label_t>
class MinMaxScaler : Scaler<Datum_t, Label_t> {
  public:
    MinMaxScaler( ) :
    range(std::pair(0, 1)) 
    { }

    MinMaxScaler( 
      const std::pair<int, int> &range // maybe floating-point?
    ) :
    range(range) {
      if(range.first >= range.second) throw std::runtime_error("Minmax range must have the (min, max) form, with min < max. Please choose a valid range.");
    }

    virtual void fit(const Dataset<Datum_t, Label_t> &dataset);

    virtual void fit(const Dataset<Datum_t, Label_t> &dataset, const std::vector<int> &fit_indices);

    virtual Dataset<Datum_t, Label_t> transform(Dataset<Datum_t, Label_t> dataset) const;

    virtual Dataset<Datum_t, Label_t> transform(const Dataset<Datum_t, Label_t> &dataset, const std::vector<int> &fit_indices) const;

    virtual Datum_t inverse_transform(Datum_t datum) const;

    std::pair<int, int> get_range( ) const { return range; }

    Datum_t get_min( ) { return min_in_dataset; }

    Datum_t get_max( ) { return max_in_dataset; }

  private:
    std::pair<int, int> range;

    Datum_t min_in_dataset;

    Datum_t max_in_dataset;

};

} // namespace tudat_learn

#include "tudat-learn/processing/scalers/minmax_scaler.tpp"

#endif // TUDAT_LEARN_MINMAX_SCALER_HPP