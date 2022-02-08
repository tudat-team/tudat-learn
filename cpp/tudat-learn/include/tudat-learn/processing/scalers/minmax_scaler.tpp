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
  if(dataset.size() < 2) throw std::runtime_error("Dataset needs to be of size 2 or greater for the MinMaxScaler use to be possible.");
  
  min_in_dataset = dataset.data_at(0);
  max_in_dataset = dataset.data_at(0);

  for(int i = 1; i < dataset.size(); ++i) {
    min_in_dataset = this->elementwise_min(min_in_dataset, dataset.data_at(i));
    max_in_dataset = this->elementwise_max(max_in_dataset, dataset.data_at(i));
  }

  difference_dataset = max_in_dataset - min_in_dataset;
}

template <typename Datum_t, typename Label_t>
void MinMaxScaler<Datum_t, Label_t>::fit(const Dataset<Datum_t, Label_t> &dataset, const std::vector<int> &fit_indices) {
  if(    dataset.size() < 2) throw std::runtime_error("Dataset needs to be of size 2 or greater for the MinMaxScaler use to be possible.");
  if(fit_indices.size() < 2) throw std::runtime_error("Fit_indices needs to be of size 2 or greater for it to be possible to fit MinMaxScaler.");

  min_in_dataset = dataset.data_at(fit_indices.at(0));
  max_in_dataset = dataset.data_at(fit_indices.at(0));

  for(int i = 1; i < fit_indices.size(); ++i) {
    min_in_dataset = this->elementwise_min(min_in_dataset, dataset.data_at(fit_indices.at(i)));
    max_in_dataset = this->elementwise_max(max_in_dataset, dataset.data_at(fit_indices.at(i)));
  }

  difference_dataset = max_in_dataset - min_in_dataset;
}

template <typename Datum_t, typename Label_t>
Dataset<Datum_t, Label_t> MinMaxScaler<Datum_t, Label_t>::transform(Dataset<Datum_t, Label_t> dataset) const {
  for(int i = 0; i < dataset.size(); ++i) {
    // Issues with eigen compiling the operator functions for rvalues make it necessary to have
    // the inputs as lvalues, hence named variables.

    Datum_t scaled_datum = dataset.data_at(i) - min_in_dataset;
            scaled_datum = this->operator_elementwise_division(scaled_datum, difference_dataset) * difference_range;
            scaled_datum = this->operator_scalar_addition(scaled_datum, range.first);

    dataset.data_at(i) = scaled_datum;
  }

  return dataset;
}

template <typename Datum_t, typename Label_t>
Dataset<Datum_t, Label_t> MinMaxScaler<Datum_t, Label_t>::transform(const Dataset<Datum_t, Label_t> &dataset, const std::vector<int> &transform_indices) const {
  Dataset<Datum_t, Label_t> out_dataset;
  out_dataset.reserve(transform_indices.size());

  for(int i = 0; i < transform_indices.size(); ++i) {    
    // Issues with eigen compiling the operator functions for rvalues make it necessary to have
    // the inputs as lvalues, hence named variables.

    Datum_t scaled_datum = dataset.data_at(transform_indices.at(i)) - min_in_dataset;
            scaled_datum = this->operator_elementwise_division(scaled_datum, difference_dataset) * difference_range;
            scaled_datum = this->operator_scalar_addition(scaled_datum, range.first);

    out_dataset.push_back(scaled_datum, dataset.labels_at(transform_indices.at(i)));
  }

  return out_dataset;
}

template <typename Datum_t, typename Label_t>
Datum_t MinMaxScaler<Datum_t, Label_t>::inverse_transform(Datum_t datum) const {
  // Issues with eigen compiling the operator functions for rvalues make it necessary to have
  // the inputs as lvalues, hence named variables.

  datum = this->operator_scalar_subtraction(datum, range.first) / difference_range;
  datum = this->operator_elementwise_multiplication(datum, difference_dataset) + min_in_dataset;

  return datum;
}

} // namespace tudat_learn

#endif // TUDAT_LEARN_MINMAX_SCALER_TPP