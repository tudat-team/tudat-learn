/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_REGRESSOR_HPP
#define TUDAT_LEARN_REGRESSOR_HPP

#include "tudat-learn/estimator.hpp"
#include "tudat-learn/dataset.hpp"

namespace tudat_learn
{
  
/**
 * @brief Base Regressor class. \n
 * Provides a base class implementation for all the regressors in tudat-learn. Receives both a Datum_t and a Label_t template
 * parameters, with the latter being defaulted to none_t, in case one is using an unlabelled Dataset.
 * 
 * @tparam Datum_t The type of a single feature vector. Can be a simple scalar, an std::vector<T>, an Eigen::Matrix, ...
 * @tparam Label_t The type of a single label. Like the Datum_t, can be a simple scalar, an std::vector<T>, an Eigen::Matrix, ...
 */
template <typename Datum_t, typename Label_t = none_t>
class Regressor : public Estimator<Datum_t, Label_t> {
  protected:
    /**
     * @brief Deleted default constructor, to make sure a shared_ptr is provided to the dataset.
     * 
     */
    Regressor( ) = delete;

    /**
     * @brief Protected constructor, ensuring it is only called from the classes that inherit from REgressor.
     * Like the Estimator, it receives a shared_ptr to a Dataset, and stores a copy of it under the dataset_ptr variable.
     * 
     * @param dataset_ptr shared_ptr to be stored under the dataset_ptr member variable.
     */
    Regressor(const std::shared_ptr< Dataset<Datum_t, Label_t> > &dataset_ptr)
    : Estimator<Datum_t, Label_t>(dataset_ptr) { }

    /**
     * @brief Virtual destructor, as the class has virtual methods.
     * 
     */
    virtual ~Regressor( ) { }

  public:

    /**
     * @brief Pure virtual method to fit the estimator to the dataset. To be implemented in the derived classes.
     * 
     */
    virtual void fit( ) = 0;

    /**
     * @brief Similar to the fit() method. Pure virtual method to fit the estimator to feature vectors at certain
     * indices in the Dataset.
     * 
     * @param fit_indices Vector with the indices of the feature vectors to which the Regressor is going to be fitted.
     */
    virtual void fit(const std::vector<size_t> &fit_indices) = 0;

};
  
} // namespace tudat_learn


#endif // TUDAT_LEARN_REGRESSOR_HPP