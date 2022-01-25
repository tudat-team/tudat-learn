/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_MINMAX_SCALER_TPP
#define TUDAT_LEARN_MINMAX_SCALER_TPP

#ifndef TUDAT_LEARN_MINMAX_SCALER_HPP
#ERROR __FILE__ should only be included from processing/scalers/minmax_scaler.hpp
#endif

namespace tudat_learn
{

template <typename Datum_t, typename Label_t>
void MinMaxScaler<Datum_t, Label_t>::fit(const Dataset<Datum_t, Label_t> &dataset) {

}

template <typename Datum_t, typename Label_t>
void MinMaxScaler<Datum_t, Label_t>::fit(const Dataset<Datum_t, Label_t> &dataset, const std::vector<int> &fit_indices) {
  
}

template <typename Datum_t, typename Label_t>
Dataset<Datum_t, Label_t> MinMaxScaler<Datum_t, Label_t>::transform(Dataset<Datum_t, Label_t> dataset) const {
  
}

template <typename Datum_t, typename Label_t>
Dataset<Datum_t, Label_t> MinMaxScaler<Datum_t, Label_t>::transform(const Dataset<Datum_t, Label_t> &dataset, const std::vector<int> &fit_indices) const {
  
}

template <typename Datum_t, typename Label_t>
Datum_t MinMaxScaler<Datum_t, Label_t>::inverse_transform(Datum_t dataset) const {
  
}

} // namespace tudat_learn

#endif // TUDAT_LEARN_MINMAX_SCALER_TPP