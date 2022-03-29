/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_CROSS_VALIDATION_TPP 
#define TUDAT_LEARN_CROSS_VALIDATION_TPP 

#ifndef TUDAT_LEARN_CROSS_VALIDATION_HPP
#ERROR __FILE__ should only be included from /model_evaluation/cross_validation.hpp
#endif

namespace tudat_learn
{

template <typename Datum_t, typename Label_t>
std::vector< std::vector<double> > CrossValidation<Datum_t, Label_t>::cross_validate( ) const {
  std::vector< std::pair< std::vector<size_t>, std::vector<size_t> > > splits(this->split_ptr->split());

  std::vector< std::vector<double> > result;
  result.reserve(splits.size());

  for(size_t i = 0u; i < splits.size(); ++i) {
    // fits the estimator to the current training set
    this->estimator_ptr->fit(splits[i].first);

    // creates the vector that will hold the metrics result for this training/validation pair
    std::vector<double> metrics_result;
    metrics_result.reserve(this->metrics.size());

    for(const auto &metric : metrics)
      metrics_result.push_back(metric(this->dataset_ptr, this->estimator_ptr, splits[i].second));

    result.push_back(metrics_result);
  }

  return result;
}

} // namespace tudat_learn


#endif // TUDAT_LEARN_CROSS_VALIDATION_TPP 