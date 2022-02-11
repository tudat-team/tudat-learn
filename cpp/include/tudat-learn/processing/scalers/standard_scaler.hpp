/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_STANDARD_SCALER_HPP
#define TUDAT_LEARN_STANDARD_SCALER_HPP

#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "tudat-learn/dataset.hpp"
#include "tudat-learn/processing/scaler.hpp"

namespace tudat_learn
{

/**
 * @brief StandardScaler. \n 
 * Scales a dataset such that each feature has a zero mean and unit standard deviation.
 * 
 * @tparam Datum_t The type of a single feature vector. Must be an arithmetic or an Eigen type.
 * @tparam Label_t The type of a single label. Can be a simple scalar, an std::vector<T>, an Eigen::Matrix, ...
 */
template <typename Datum_t, typename Label_t>
class StandardScaler : Scaler<Datum_t, Label_t> {
  public:

    /**
     * @brief Default constructor.
     * 
     * @tparam Datum_tt Same as Datum_t
     * @tparam std::enable_if_t< is_eigen<Datum_tt>::value || std::is_arithmetic<Datum_tt>::value> The constructor is only
     * enabled if Datum_tt is of an arithmetic or Eigen type.
     */
    template <
      typename Datum_tt = Datum_t,
      typename = std::enable_if_t< is_eigen<Datum_tt>::value || std::is_arithmetic<Datum_tt>::value>
    >
    StandardScaler( ) : Scaler<Datum_t, Label_t>() { }

    /**
     * @brief Virtual destructor, as the class has virtual functions.
     * 
     */
    virtual ~StandardScaler( ) { }

    /**
     * @brief Fits the StandardScaler to the Dataset. \n
     * \n Extracts the feature-wise mean, standard deviation and variance in the dataset. Saves them under mean,
     * standard_deviation, and variance, respectively.
     * 
     * @param dataset Constant reference to the Dataset object.
     */
    virtual void fit(const Dataset<Datum_t, Label_t> &dataset) override;

    /**
     * @brief Fits the StandardScaler to feature vectors at specific indices in the Dataset. \n
     * Extracts the feature-wise mean, standard deviation and variance in the dataset. Saves them under mean,
     * standard_deviation, and variance, respectively.
     * 
     * @param dataset Constant reference to the Dataset object.
     * @param fit_indices Constant reference to the vector with the indices to which the MinMaxScaler is going to be fitted.
     */
    virtual void fit(const Dataset<Datum_t, Label_t> &dataset, const std::vector<int> &fit_indices) override;

    /**
     * @brief Scales the dataset the dataset according to the information obtained with the fit method.
     * Essentially for every datum, new_datum = (datum - mean) / standard_deviation;
     * 
     * @param dataset Copy of the Dataset used to return the scaled dataset.
     * @return Dataset<Datum_t, Label_t> Scaled dataset.
     */
    virtual Dataset<Datum_t, Label_t> transform(Dataset<Datum_t, Label_t> dataset) const override;

    /**
     * @brief Scales the dataset according to the information obtained with the fit method, and only scales feature vectors
     * whose indices are contained in the fit_indices vector, outputting a new Dataset with those feature vectors, but scaled.
     * Essentially for every relevant datum, new_datum = (datum - mean) / standard_deviation;
     * 
     * @param dataset Constant reference to the Dataset that contains the feature vectors being scaled.
     * @param fit_indices Vector with the indices of the feature vectors that are going to be scaled.
     * @return Dataset<Datum_t, Label_t> New dataset with the chosen feature vectors scaled.
     */
    virtual Dataset<Datum_t, Label_t> transform(const Dataset<Datum_t, Label_t> &dataset, const std::vector<int> &fit_indices) const override;

    /**
     * @brief Applies the transformation inverse to scaling, obtained with transform.
     * Essentially, new_datum = (datum + mean) * standard_deviation.
     * 
     * @param datum Scaled feature vector meant to be returned to the original scaled.
     * @return Datum_t Feature vector at the original scale.
     */
    virtual Datum_t inverse_transform(Datum_t datum) const override;

    /**
     * @brief Gets the feature-wise mean.
     * 
     * @return Datum_t eature-wise mean.
     */
    Datum_t get_mean( )               const { return mean; }

    /**
     * @brief Gets the standard feature-wise deviation.
     * 
     * @return Datum_t Feature-wise standard deviation.
     */
    Datum_t get_standard_deviation( ) const { return standard_deviation; }

    /**
     * @brief Gets the feature-wise variance.
     * 
     * @return Datum_t Feature-wise variance.
     */
    Datum_t get_variance( )           const { return variance; }

  protected:

    /**
     * @brief Performs an iteration of the Welford's algorithm to compute a running standard deviation in a single loop.
     * https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
     * 
     * @tparam T Type of the data that will be normalised.
     * @param count Reference to the number of feature vectors that have been processed.
     * @param mean  Reference to the running mean.
     * @param m2    Reference to the running M2 value.
     * @param new_value Constant reference to the new feature vector being processed.
     * @return std::enable_if< std::is_same<T, Datum_t>::value || std::is_same<T, Label_t>::value,
     * void >::type Returns void.
     */
    template <typename T>
    typename std::enable_if< std::is_same<T, Datum_t>::value || std::is_same<T, Label_t>::value, 
    void >::type welford_iteration(int &count, T &mean, T &m2, const T &new_value);

    


  protected:
    Datum_t mean;               /**< Feature-wise mean. */

    Datum_t standard_deviation; /**< Feature-wise standard deviation. */

    Datum_t variance;           /**< Feature-wise variance. */
};


} // namespace tudat_learn

#include "tudat-learn/processing/scalers/standard_scaler.tpp"

#endif // TUDAT_LEARN_STANDARD_SCALER_HPP