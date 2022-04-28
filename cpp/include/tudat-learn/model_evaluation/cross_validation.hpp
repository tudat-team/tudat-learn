/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_CROSS_VALIDATION_HPP 
#define TUDAT_LEARN_CROSS_VALIDATION_HPP 

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "tudat-learn/dataset.hpp"
#include "tudat-learn/estimator.hpp"
#include "tudat-learn/model_evaluation/split.hpp"

namespace tudat_learn
{

/**
 * @brief Implements a general cross-validation procedure. Different types of splits can be used, such as a k-fold
 * split, to perform k-fold cross-validation, or a simple train-validation holdout split.
 * Additionally, different metrics can be implemented. These are simple functions for now, instead of tudat-learn
 * classes. Examples can be found in the cross_validation_test.cpp unit test.
 * 
 * @tparam Datum_t The type of a single feature vector. Can be a simple scalar, an std::vector<T>, an Eigen::Matrix, ...
 * @tparam Label_t The type of a single label. Like the Datum_t, can be a simple scalar, an std::vector<T>, an Eigen::Matrix, ...
 */
template <typename Datum_t, typename Label_t>
class CrossValidation {
  public:
    /**
     * @brief Deleted default constructor, to ensure CrossValidation is created with settings
     * 
     */
    CrossValidation( ) = delete;

    /**
     * @brief Construct a new CrossValidation object with the specified settings.
     * 
     * @param dataset_ptr Shared pointer to a Dataset with which the Estimator is trained.
     * @param estimator_ptr Shared pointer to an Estimator.
     * @param split_ptr Shared pointer to a Split
     * @param metrics Vector of functions that take as inputs a shared pointer to a Dataset, another to an
     * Estimator and finally a vector of size_t types, to identify the indices of the Dataset at which the
     * model's performance is to be evaluated.
     */
    CrossValidation(
      const std::shared_ptr< Dataset<Datum_t, Label_t> > &dataset_ptr,
      const std::shared_ptr< Estimator<Datum_t, Label_t> > &estimator_ptr,
      const std::shared_ptr< Split<Datum_t, Label_t> > &split_ptr,
      const std::vector< std::function<double(
        const std::shared_ptr<Dataset<Datum_t, Label_t>> &,
        const std::shared_ptr<Estimator<Datum_t, Label_t>> &, 
        const std::vector<size_t> &
      )> > &metrics
    ) :
    dataset_ptr(dataset_ptr),
    estimator_ptr(estimator_ptr),
    split_ptr(split_ptr),
    metrics(metrics)
    { }

    /**
     * @brief Function that performs cross-validation on the estimator.
     * 
     * @return std::vector< std::vector<double> > Vector of vector<double>. The number of vector<double> corresponds to the number
     * of splits, while the number of doubles in each of those vectors corresponds to the number of metrics.
     */
    std::vector< std::vector<double> > cross_validate( ) const;

  protected:
    std::shared_ptr< Dataset<Datum_t, Label_t> > dataset_ptr;       /**< Shared pointer to the Dataset. */

    std::shared_ptr< Estimator<Datum_t, Label_t> > estimator_ptr;   /**< Shared pointer to the Estimator. */

    std::shared_ptr< Split<Datum_t, Label_t> > split_ptr;           /**< Shared poitner to the Split. */

    std::vector< std::function<double(
      const std::shared_ptr<Dataset<Datum_t, Label_t>> &,
      const std::shared_ptr<Estimator<Datum_t, Label_t>> &, 
      const std::vector<size_t> &
    )> > metrics;                                                   /**< Vector of the functions used as metrics during the cross-validation. */
};

} // namespace tudat_learn


#include "tudat-learn/model_evaluation/cross_validation.tpp"

#endif // TUDAT_LEARN_CROSS_VALIDATION_HPP 