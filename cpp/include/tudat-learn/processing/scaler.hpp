/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_SCALER_HPP
#define TUDAT_LEARN_SCALER_HPP

#include <vector>
#include <type_traits>

#include <Eigen/Core>

#include "tudat-learn/dataset.hpp"
#include "tudat-learn/operator.hpp"
#include "tudat-learn/processing.hpp"

namespace tudat_learn
{

/**
 * @brief Base Scaler class. \n 
 * Provides a base class implementation for all the scalers in tudat-learn. Receives both a Datum_t and a Label_t template
 * parameters, with the latter being defaulted to none_t, in case one is using an unlabelled Dataset. \n 
 * Inherits from Processing and Operator.
 * 
 * @tparam Datum_t The type of a single feature vector. Can be a simple scalar, an std::vector<T>, an Eigen::Matrix, ...
 * @tparam Label_t The type of a single label. Like the Datum_t, can be a simple scalar, an std::vector<T>, an Eigen::Matrix, ...
 */
template <typename Datum_t, typename Label_t = none_t>
class Scaler : public Processing<Datum_t, Label_t>, public Operator<Datum_t>{
  protected:
    /**
     * @brief Protected default constructor, ensuring it is only called from classes that derive from Scaler.
     * 
     */
    Scaler() : Processing<Datum_t,Label_t>(), Operator<Datum_t>() { }
  
  public:

    /**
     * @brief Virtual destructor, as the class has virtual functions.
     * 
     */
    virtual ~Scaler( ) { }

    /**
     * @brief Pure virtual method to fit the scaler to the Dataset. Can be used to extract some charecteristics from the Dataset
     * object, such as the maximum/minimum values, or the average value.
     * 
     * @param dataset Constant reference to the Dataset object.
     */
    virtual void fit(const Dataset<Datum_t, Label_t> &dataset) = 0;

    /**
     * @brief Pure virtual method to fit the scaler to the Dataset. However, only fits the Scaler to feature vectors at
     * certain indices in the Dataset. Can be used to extract some charecteristics from the Dataset object, such as the
     * maximum/minimum values, or the average value.
     * 
     * @param dataset Constant reference to the Dataset object.
     * @param fit_indices Vector with the indices of the feature vectors to which the Scaler is going to be fitted.
     */
    virtual void fit(const Dataset<Datum_t, Label_t> &dataset, const std::vector<int> &fit_indices) = 0;

    /**
     * @brief Pure virtual method that scales the dataset according to the information obtained with the fit method.
     * 
     * @param dataset Copy of the Dataset used to return the scaled dataset.
     * @return Dataset<Datum_t, Label_t> Scaled dataset.
     */
    virtual Dataset<Datum_t, Label_t> transform(Dataset<Datum_t, Label_t> dataset) const = 0;

    /**
     * @brief Pure virtual method that scales the dataset according to the information obtained with the fit method, and only scales feature vectors
     * whose indices are contained in the fit_indices vector, outputting a new Dataset with those feature vectors, but scaled.
     * 
     * @param dataset Constant reference to the Dataset that contains the feature vectors being scaled.
     * @param fit_indices Vector with the indices of the feature vectors that are going to be scaled.
     * @return Dataset<Datum_t, Label_t> New dataset with the chosen feature vectors scaled.
     */
    virtual Dataset<Datum_t, Label_t> transform(const Dataset<Datum_t, Label_t> &dataset, const std::vector<int> &fit_indices) const = 0;

    // virtual void fit_transform(Dataset<Datum_t, Label_t> &dataset) const = 0;

    /**
     * @brief Pure virtual method that applies the transformation inverse to scaling, obtained with transform.
     * 
     * @param datum Scaled feature vector meant to be returned to the original scaled.
     * @return Datum_t Feature vector at the original scale.
     */
    virtual Datum_t inverse_transform(Datum_t datum) const = 0;

  protected:
    
};

} // namespace tudat_learn

#endif // TUDAT_LEARN_SCALER_HPP