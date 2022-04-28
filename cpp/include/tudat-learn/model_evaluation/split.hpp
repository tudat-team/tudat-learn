/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_SPLIT_HPP 
#define TUDAT_LEARN_SPLIT_HPP 

#include <memory>
#include <vector>

namespace tudat_learn
{

/**
 * @brief Base Split class. \n 
 * Provides a base class implementation for all the splits in tudat-learn. Focuses on providing vector<pair<vector,vector>> splits,
 * where the outmost vector contains a pair for each split that was done (for instance, k pairs in k-fold cross-validation). The
 * first inner vector contains the indices at which the data used for training will be within the dataset, whereas the second inner
 * vector contains the indices corresponding to the validation data. This prevents copies of said data from being made.
 * Receives template parameters since there are splits that are dependent on the type of data that one is splitting, for instance,
 * for classification when there is the need that a k-fold split maintains the label distribution in each of the folds.
 * 
 * @tparam Datum_t The type of a single feature vector. Can be a simple scalar, an std::vector<T>, an Eigen::Matrix, ...
 * @tparam Label_t The type of a single label. Like the Datum_t, can be a simple scalar, an std::vector<T>, an Eigen::Matrix, ...
 */
template <typename Datum_t, typename Label_t>
class Split {
  protected:
    /**
     * @brief Protected default constructor, ensuring it is only called from classes that derive from Split.
     * 
     * @param shuffle Flag that signals whether the dataset is to be shuffled before the splits are performed.
     */
    Split(
      const std::shared_ptr< Dataset<Datum_t, Label_t> > &dataset_ptr,
      const bool shuffle=false
      ) : 
      dataset_ptr(dataset_ptr),
      shuffle(shuffle) 
      { }

  public:
    /**
     * @brief Virtual destructor, as the class has virtual functions.
     * 
     */
    virtual ~Split( ) { }

    /**
     * @brief Pure virtual method that makes test/validation splits using the whole dataset.
     * 
     * @return std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> Vector of pairs of vectors of indices. There are
     * k pairs, where k is the number of times the dataset is split. The first vector in the pair contains the indices of the data
     * within the dataset with which the estimator will be trained with, while the second vector contains indices pertaining to the
     * data to be used in validation: 1st vector in the pair - training data; 2nd vector in the pair - validation data.
     */
    virtual std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> split( ) const = 0;

    

    // /**
    //  * @brief Pure virtual method that takes a vector of indices from which the split is made as an input, which is useful if 
    //  * one does not want to use the whole dataset.
    //  * 
    //  * @param valid_indices Vector of indices from which the split is to be made.
    //  * @return std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> Vector of pairs of vectors of indices. There are
    //  * k pairs, where k is the number of times the dataset is split. The first vector in the pair contains the indices of the data
    //  * within the dataset with which the estimator will be trained with, while the second vector contains indices pertaining to the
    //  * data to be used in validation: 1st vector in the pair - training data; 2nd vector in the pair - validation data.
    //  */
    // virtual std::vector<std::pair< std::vector<size_t>, std::vector<size_t> >> split(const std::vector<size_t> &valid_indices ) const = 0;

  protected:
    std::shared_ptr< Dataset<Datum_t, Label_t> > dataset_ptr;

    bool shuffle; /**< Flag that signals whether the dataset is to be shuffled before the splits are performed. */    

};

} // namespace tudat_learn

#endif // TUDAT_LEARN_SPLIT_HPP 