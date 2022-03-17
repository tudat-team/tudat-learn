/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_K_FOLD_SPLIT_TPP 
#define TUDAT_LEARN_K_FOLD_SPLIT_TPP 

#ifndef TUDAT_LEARN_K_FOLD_SPLIT_HPP
#ERROR __FILE__ should only be included from /model_evaluation/splits/k_fold_split.hpp
#endif

namespace tudat_learn
{

template <typename Datum_t, typename Label_t>
std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> KFoldSplit<Datum_t, Label_t>::split( ) const {
  // creates a vector of indices from 0 to dataset.size - 1
  std::vector<size_t> indices(this->dataset_ptr->size());

  std::iota(indices.begin(), indices.end(), 0);
  // shuffles the indices if the shuffle flag is set to true
  if(this->shuffle)
    std::shuffle(indices.begin(), indices.end(), Random::get_rng());

  // computes the floating-point size of each of the folds (will be rounded down as the folds are computed)
  float step = static_cast<float>(this->dataset_ptr->size()) / static_cast<float>(n_folds);

  std::vector<size_t> fold_limits;
  fold_limits.reserve(n_folds + 1);

  // computes the indices at which the folds start and end. E.g., size = 10, n_folds = 4
  // fold_limits = 0 2 5 7 10, each of the 4 folds starts at index fold_limits(i) and ends at index (fold_limits(i + 1) - 1)
  for(size_t k = 0u; k <= n_folds; ++k)
    fold_limits.push_back(static_cast<size_t>(k * step));

  std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> folds_vector;
  folds_vector.reserve(n_folds);

  for(size_t k = 0u; k < n_folds; ++k) {
    size_t n_validation_instances = fold_limits.at(k+1) - fold_limits.at(k);

    std::vector<size_t> training_instances;
    std::vector<size_t> validation_instances;
    training_instances.reserve(this->dataset_ptr->size() - n_validation_instances);
    validation_instances.reserve(n_validation_instances);

    // getting training data indices
    std::copy(indices.begin(), indices.begin() + fold_limits.at(k), std::back_inserter(training_instances));
    std::copy(indices.begin() + fold_limits.at(k+1), indices.end(), std::back_inserter(training_instances));

    // getting validation data indices
    std::copy(indices.begin() + fold_limits.at(k), indices.begin() + fold_limits.at(k+1), std::back_inserter(validation_instances));

    folds_vector.push_back(std::make_pair(std::move(training_instances), std::move(validation_instances)));
  }

  return folds_vector;  
}


} // namespace tudat_learn

#endif // TUDAT_LEARN_K_FOLD_SPLIT_TPP