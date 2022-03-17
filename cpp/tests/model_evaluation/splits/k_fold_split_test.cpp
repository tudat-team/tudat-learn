/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#include <iostream>
#include <memory>
#include <unordered_set>
#include <utility>
#include <vector>

#include <tudat-learn/dataset.hpp>
#include <tudat-learn/model_evaluation/splits/k_fold_split.hpp>
#include <tudat-learn/random.hpp>

void print_splits(const std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> &splits) {
  for(size_t i = 0u; i < splits.size(); ++i) {
    std::cout << "Training Data at split " << i << ":" << std::endl;
    for(const auto &train_index : splits[i].first)
      std::cout << train_index << ", ";
    std::cout << std::endl;

    std::cout << "Validation Data at split " << i << ":" << std::endl;
    for(const auto &validation_index : splits[i].second)
      std::cout << validation_index << ", ";
    std::cout << std::endl;
  }
}

bool compare_splits(const std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> &splits1, const std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> &splits2) {
  // compares number of splits
  if(splits1.size() != splits2.size())
    return false;

  for(size_t i = 0; i < splits1.size(); ++i) {
    // compares number of training samples in the split
    if(splits1[i].first.size() != splits2[i].first.size())
      return false;
    // compares number of validation samples in the split
    if(splits1[i].second.size() != splits2[i].second.size())
      return false;

    // compares indices of training data individually
    for(size_t j = 0; j < splits1[i].first.size(); ++j)
      if(splits1[i].first[j] != splits2[i].first[j]) {
        std::cout << "Training data different on fold " << i << " at index " << j << "." << std::endl;
        return false;
      }

    // compares indices of validation data individually
    for(size_t j = 0; j < splits1[i].second.size(); ++j)
      if(splits1[i].second[j] != splits2[i].second[j]) {
        std::cout << "Validation data different on fold " << i << " at index " << j << "." << std::endl;
        return false;
      }
  }

  return true;
}

bool are_valid_k_fold_splits(const std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> &splits) {
  size_t dataset_size = splits[0].first.size() + splits[0].second.size();
  // check if all the folds have the same total size
  for(const auto &fold: splits)
    if(fold.first.size() + fold.second.size() != dataset_size)
      return false;

  std::unordered_set<size_t> validation_indices;

  for(size_t i = 0u; i < splits.size(); ++i) {
    std::unordered_set<size_t> current_train_indices;
    std::unordered_set<size_t> current_validation_indices;

    for(size_t j = 0u; j < splits[i].first.size(); ++j)
      current_train_indices.insert(splits[i].first[j]);

    for(size_t j = 0u; j < splits[i].second.size(); ++j) {
      current_validation_indices.insert(splits[i].second[j]);
      if(validation_indices.find(splits[i].second[j]) == validation_indices.end())
        validation_indices.insert(splits[i].second[j]);
      else
        return false;
    }

    if(current_train_indices.size() != splits[i].first.size())
      return false;

    if(current_validation_indices.size() != splits[i].second.size())
      return false;

    if(current_train_indices.size() + current_validation_indices.size() != dataset_size)
      return false;
  }

  if(validation_indices.size() != dataset_size)
    return false;

  return true; 
}

int main( ) {

  // dataset type does not matter
  auto dataset_ptr = std::make_shared< tudat_learn::Dataset<int, int> >();

  size_t dataset_size = 10;
  size_t n_folds = 4;

  for(size_t i = 0u; i < dataset_size; ++i)
    dataset_ptr->push_back(static_cast<int>(i), static_cast<int>(i));
    
  // f_fold_split_(dataset size)_(n_folds)
  tudat_learn::KFoldSplit<int, int> k_fold_split_10_4(
   dataset_ptr,
   n_folds,
   false 
  );

  // expected_(dataset size)_(n_folds)
  std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> expected_10_4({
    {{2,3,4,5,6,7,8,9},{0,1}},
    {{0,1,5,6,7,8,9},{2,3,4}},
    {{0,1,2,3,4,7,8,9},{5,6}},
    {{0,1,2,3,4,5,6},{7,8,9}}
  });

  if(!compare_splits(k_fold_split_10_4.split(), expected_10_4))
    return 1;

  std::cout << "Splits for a dataset of size 10 and 4 folds:" << std::endl;
  print_splits(k_fold_split_10_4.split());

  dataset_size = 15;
  n_folds = 10;

  for(size_t i = dataset_ptr->size(); i < dataset_size; ++i)
    dataset_ptr->push_back(static_cast<int>(i), static_cast<int>(i));

  // f_fold_split_(dataset size)_(n_folds)
  tudat_learn::KFoldSplit<int, int> k_fold_split_15_10(
   dataset_ptr,
   n_folds,
   false 
  );

  // expected_(dataset size)_(n_folds)
  std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> expected_15_10({
    {{1,2,3,4,5,6,7,8,9,10,11,12,13,14},{0}},
    {{0,3,4,5,6,7,8,9,10,11,12,13,14},{1,2}},
    {{0,1,2,4,5,6,7,8,9,10,11,12,13,14},{3}},
    {{0,1,2,3,6,7,8,9,10,11,12,13,14},{4,5}},
    {{0,1,2,3,4,5,7,8,9,10,11,12,13,14},{6}},
    {{0,1,2,3,4,5,6,9,10,11,12,13,14},{7,8}},
    {{0,1,2,3,4,5,6,7,8,10,11,12,13,14},{9}},
    {{0,1,2,3,4,5,6,7,8,9,12,13,14},{10,11}},
    {{0,1,2,3,4,5,6,7,8,9,10,11,13,14},{12}},
    {{0,1,2,3,4,5,6,7,8,9,10,11,12},{13,14}},
  });
  
  if(!compare_splits(k_fold_split_15_10.split(), expected_15_10))
    return 1;

  std::cout << "Splits for a dataset of size 15 and 10 folds:" << std::endl;
  print_splits(k_fold_split_15_10.split());

  
  // testing with shuffle
  tudat_learn::Random::set_seed(0);
  tudat_learn::KFoldSplit<int, int> k_fold_split_shuffle_15_10(
   dataset_ptr,
   n_folds,
   true
  );

  auto shuffle_split_15_10 = k_fold_split_shuffle_15_10.split();
  if(!are_valid_k_fold_splits(shuffle_split_15_10))
    return false;

  std::cout << "Splits for a dataset of size 15 and 10 folds, with shuffle:" << std::endl;
  print_splits(shuffle_split_15_10);

  
  // testing multiple fold sizes with shuffle
  dataset_size = 1000;

  for(size_t i = dataset_ptr->size(); i < dataset_size; ++i)
    dataset_ptr->push_back(static_cast<int>(i), static_cast<int>(i));


  size_t repetitions = 100;
  for(size_t k = 1; k <= repetitions; ++k) {
    tudat_learn::KFoldSplit<int, int> k_fold_split_loop(
    dataset_ptr,
    n_folds,
    true 
    );

    if(!are_valid_k_fold_splits(k_fold_split_loop.split()))
      return 1;
  }

  // testing are_valid_k_fold_splits
 std::vector< std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> > wrong_examples({
    {
      {{2,3,4,5,6,7,8,9},{0,1}},
      {{0,1,5,6,7,8,9},{2,3,4}},
      {{0,1,2,3,4,7,8,9},{5,6}},
      {{0,1,2,3,4,5,6},{7,8}}
    },
    {
      {{2,3,4,5,6,7,8,9},{0,1}},
      {{0,1,5,6,7,8,9},{2,3,4}},
      {{0,1,2,3,4,7,8,9},{5,6}},
      {{0,1,2,3,4,5,6},{7,8,9}},
      {{0,1,2,3,4,5,6},{7,8,9}}
    },
  });

  for(const auto &it: wrong_examples)
    if(are_valid_k_fold_splits(it))
      return 1;


  return 0;
}