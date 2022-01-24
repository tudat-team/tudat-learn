/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_STANDARD_SCALER_TPP
#define TUDAT_LEARN_STANDARD_SCALER_TPP

#ifndef TUDAT_LEARN_STANDARD_SCALER_HPP
#ERROR __FILE__ should only be included from processing/scalers/standard_scaler.hpp
#endif

namespace tudat_learn
{

template <typename Datum_t, typename Label_t>
void StandardScaler<Datum_t, Label_t>::fit(const Dataset<Datum_t, Label_t> &dataset) {
      
      if(dataset.size() < 2) throw std::runtime_error("Dataset needs to be of size 2 or greater for the StandardScaler use to be possible.");

      // WELFORD'S ALGORITHM
      // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
      
      // First iteration of the Welford's Algorithm is done outside of the loop, in case the type T
      // is an Eigen type.
      int count = 1;
      Datum_t running_mean = dataset.data_at(0);
      Datum_t m2 = dataset.data_at(0) - dataset.data_at(0);
      
      for(int i = 1; i < dataset.size(); ++i) 
        welford_iteration(count, running_mean, m2, dataset.data_at(i));
      mean = running_mean;
      variance = m2 / count;
      standard_deviation = this->square_root(variance);
}

template <typename Datum_t, typename Label_t>
void StandardScaler<Datum_t, Label_t>::fit(const Dataset<Datum_t, Label_t> &dataset, const std::vector<int> &fit_indices) {
      if(    dataset.size() < 2) throw std::runtime_error("Dataset needs to be of size 2 or greater for the StandardScaler use to be possible.");
      if(fit_indices.size() < 2) throw std::runtime_error("Fit_indices needs to be of size 2 or greater for it to be possible to fit StandardScaler.");
      
      // WELFORD'S ALGORITHM
      // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
      
      // First iteration of the Welford's Algorithm is done outside of the loop, in case the type T
      // is an Eigen type.
      int count = 1;
      Datum_t running_mean = dataset.data_at(fit_indices.at(0));
      Datum_t m2 = dataset.data_at(fit_indices.at(0)) - dataset.data_at(fit_indices.at(0));
      
      for(int i = 1; i < fit_indices.size(); ++i) 
        welford_iteration(count, running_mean, m2, dataset.data_at(fit_indices.at(i)));
      mean = running_mean;
      variance = m2 / count;
      standard_deviation = this->square_root(variance);
}

template <typename Datum_t, typename Label_t> template <typename T>
typename std::enable_if< std::is_same<T, Datum_t>::value || std::is_same<T, Label_t>::value, 
void >::type StandardScaler<Datum_t, Label_t>::welford_iteration(int &count, T &mean, T &m2, const T &new_value) {
  count++;
  T delta = new_value - mean;
  mean += delta / count;
  T delta2 = new_value - mean;
  m2 += this->operator_multiply_elementwise(delta, delta2);
}


template <typename Datum_t, typename Label_t>
Dataset<Datum_t, Label_t> StandardScaler<Datum_t, Label_t>::transform(Dataset<Datum_t, Label_t> dataset) const {
  for(int i = 0; i < dataset.size(); ++i) {
    // Issues with eigen compiling the elementwise divide operator function for rvalues make it necessary to have
    // the inputs as lvalues, hence named variables.
    Datum_t mean_centered_datum = dataset.data_at(i) - mean; 
    dataset.data_at(i) = this->operator_divide_elementwise(mean_centered_datum, standard_deviation); 
  }

  return dataset;
}

template <typename Datum_t, typename Label_t>
Dataset<Datum_t, Label_t> StandardScaler<Datum_t, Label_t>::transform(const Dataset<Datum_t, Label_t> &dataset, const std::vector<int> &transform_indices) const {
  Dataset<Datum_t, Label_t> out_dataset;
  out_dataset.reserve(transform_indices.size());

  for(int i = 0; i < transform_indices.size(); ++i) {
    // Issues with eigen compiling the elementwise divide operator function for rvalues make it necessary to have
    // the inputs as lvalues, hence named variables.
    std::cout << "Test" << i << std::endl;
    Datum_t mean_centered_datum = dataset.data_at(transform_indices.at(i)) - mean; 
    out_dataset.push_back(this->operator_divide_elementwise(mean_centered_datum, standard_deviation), dataset.labels_at(transform_indices.at(i))); 
  }

  return out_dataset;
}


} // namespace tudat_learn

#endif // TUDAT_LEARN_STANDARD_SCALER_TPP