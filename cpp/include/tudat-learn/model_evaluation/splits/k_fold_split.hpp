/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_K_FOLD_SPLIT_HPP 
#define TUDAT_LEARN_K_FOLD_SPLIT_HPP 

#include <algorithm>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tudat-learn/dataset.hpp"
#include "tudat-learn/model_evaluation/split.hpp"
#include "tudat-learn/random.hpp"

namespace tudat_learn
{

/**
 * @brief K-Fold Split. \n
 * Splits the Dataset into k folds. Each of those folds is returned as the validation dataset with the
 * remaining k-1 folds being returned as the training dataset for the current split. Naturally, results
 * in k different splits. 
 * 
 * @tparam Datum_t The type of a single feature vector. Can be a simple scalar, an std::vector<T>, an Eigen::Matrix, ...
 * @tparam Label_t The type of a single label. Like the Datum_t, can be a simple scalar, an std::vector<T>, an Eigen::Matrix, ...
 */
template <typename Datum_t, typename Label_t= none_t>
class KFoldSplit : public Split<Datum_t, Label_t> {
  public:
     
    /**
     * @brief Deleted default constructor, to ensure CrossValidation is created with settings
     * 
     */ 
    KFoldSplit( ) = delete;

    /**
     * @brief Construct a new KFoldSplit object with the specified settings.
     * 
     * @param dataset_ptr Shared pointer to the dataset to be split.
     * @param n_folds Number of folds used in the split. This number has to be smaller than or equal
     * to the size of the dataset.
     * @param shuffle Boolean flag to indeicate whther the dataset should be shuffled before the split.
     * Note that the dataset object is not actually shuffled, but the instances for each of the folds
     * are selected as if it were.
     */
    KFoldSplit(
      const std::shared_ptr< Dataset<Datum_t, Label_t> > &dataset_ptr,
      const size_t n_folds,
      const bool shuffle=false
    ) :
    Split<Datum_t, Label_t>(dataset_ptr, shuffle),
    n_folds(n_folds)
    { }

    /**
     * @brief Virtual destructor, as the class has virtual functions.
     * 
     */
    virtual ~KFoldSplit( ) { }

    /**
     * @brief Pure virtual method that makes test/validation splits using the whole dataset.
     * 
     * @return std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> Vector of pairs of vectors of indices. There are
     * k pairs, where k is the number of times the dataset is split. The first vector in the pair contains the indices of the data
     * within the dataset with which the estimator will be trained with, while the second vector contains indices pertaining to the
     * data to be used in validation: 1st vector in the pair - training data; 2nd vector in the pair - validation data.
     */
    virtual std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> split( ) const override;

    

  protected:
    size_t n_folds;   /**< Number of folds. */


};

} // namespace tudat_learn

#include "tudat-learn/model_evaluation/splits/k_fold_split.tpp"

#endif // TUDAT_LEARN_K_FOLD_SPLIT_HPP 