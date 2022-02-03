/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_STANDARD_SCALER_HPP
#define TUDAT_LEARN_STANDARD_SCALER_HPP

#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "tudat-learn/dataset.hpp"
#include "tudat-learn/processing/scaler.hpp"

namespace tudat_learn
{

template <typename Datum_t, typename Label_t>
class StandardScaler : Scaler<Datum_t, Label_t> {
  public:

    template <
      typename Datum_tt = Datum_t,
      typename = std::enable_if_t< is_eigen<Datum_tt>::value || std::is_arithmetic<Datum_tt>::value>
    >
    StandardScaler( ) : Scaler<Datum_t, Label_t>() { }

    virtual void fit(const Dataset<Datum_t, Label_t> &dataset);

    virtual void fit(const Dataset<Datum_t, Label_t> &dataset, const std::vector<int> &fit_indices);

    virtual Dataset<Datum_t, Label_t> transform(Dataset<Datum_t, Label_t> dataset) const;

    virtual Dataset<Datum_t, Label_t> transform(const Dataset<Datum_t, Label_t> &dataset, const std::vector<int> &fit_indices) const;

    virtual Datum_t inverse_transform(Datum_t datum) const;

    Datum_t get_mean( )               const { return mean; }
    Datum_t get_standard_deviation( ) const { return standard_deviation; }
    Datum_t get_variance( )           const { return variance; }

  protected:
    template <typename T>
    typename std::enable_if< std::is_same<T, Datum_t>::value || std::is_same<T, Label_t>::value, 
    void >::type welford_iteration(int &count, T &mean, T &m2, const T &new_value);

    


  protected:
    Datum_t mean;

    Datum_t standard_deviation;

    Datum_t variance;
};


} // namespace tudat_learn

#include "tudat-learn/processing/scalers/standard_scaler.tpp"

#endif // TUDAT_LEARN_STANDARD_SCALER_HPP