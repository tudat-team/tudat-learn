/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_MINMAX_SCALER_HPP
#define TUDAT_LEARN_MINMAX_SCALER_HPP

#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "tudat-learn/types.hpp"
#include "tudat-learn/processing/scaler.hpp"

namespace tudat_learn
{

/**
 * @brief Minmax Scaler. \n 
 * Scales every feature in the dataset for it to be within a prespecified range. All features are scaled to be in the same range
 * which defaults to [0,1].
 * 
 * @tparam Datum_t The type of a single feature vector. Must be an arithmetic or an Eigen type.
 * @tparam Label_t The type of a single label. Can be a simple scalar, an std::vector<T>, an Eigen::Matrix, ...
 */
template <typename Datum_t, typename Label_t>
class MinMaxScaler : Scaler<Datum_t, Label_t> {
  public:
    /**
     * @brief Default constructor. \n
     * Sets the range to be [0,1].
     * 
     * @tparam Datum_tt Same as Datum_t
     * @tparam std::enable_if_t< is_eigen<Datum_tt>::value || std::is_arithmetic<Datum_tt>::value> The constructor is only 
     * enabled if Datum_tt is of an arithmetic or Eigen type.
     */
    template <
      typename Datum_tt = Datum_t,
      typename = std::enable_if_t< is_eigen<Datum_tt>::value || std::is_arithmetic<Datum_tt>::value>
    >
    MinMaxScaler( ) :
    Scaler<Datum_t, Label_t>(), range(std::pair(0, 1)), difference_range(1) 
    { }

    /**
     * @brief Constructs an object with a specified range.
     * 
     * @tparam Datum_tt Same as Datum_t
     * @tparam std::enable_if_t< is_eigen<Datum_tt>::value || std::is_arithmetic<Datum_tt>::value> The constructor is only
     * enabled if Datum_tt is of an arithmetic or Eigen type.
     * @param range Constant reference to the user-provided range.
     */
    template <
      typename Datum_tt = Datum_t,
      typename = std::enable_if_t< is_eigen<Datum_tt>::value || std::is_arithmetic<Datum_tt>::value>
    >
    MinMaxScaler( 
      const std::pair<int, int> &range // maybe floating-point?
    ) :
    Scaler<Datum_t, Label_t>(), range(range) {
      if(range.first >= range.second) throw std::runtime_error("Minmax range must have the (min, max) form, with min < max. Please choose a valid range.");
      difference_range = range.second - range.first;
    }
    
    /**
     * @brief Virtual destructor, as the class has virtual functions.
     * 
     */
    virtual ~MinMaxScaler( ) { }

    /**
     * @brief Fits the MinMaxScaler to the Dataset. \n
     * Extracts the feature-wise maximum and minimum in the dataset. Saves them under max_in_dataset and min_in_dataset, 
     * respectively. Also computes the difference_dataset, which is the maximum with the minimum subtracted from it.
     * 
     * @param dataset Constant reference to the Dataset object.
     */
    virtual void fit(const Dataset<Datum_t, Label_t> &dataset) override;

    /**
     * @brief Fits the MinMaxScaler to feature vectors at specific indices in the Dataset. \n
     * Extracts the feature-wise maximum and minimum in the dataset. Saves them under max_in_dataset and min_in_dataset, 
     * respectively. Also computes the difference_dataset, which is the maximum with the minimum subtracted from it.
     * 
     * @param dataset Constant reference to the Dataset object.
     * @param fit_indices Constant reference to the vector with the indices to which the MinMaxScaler is going to be fitted.
     */
    virtual void fit(const Dataset<Datum_t, Label_t> &dataset, const std::vector<int> &fit_indices) override;

    /**
     * @brief Scales the dataset the dataset according to the information obtained with the fit method.
     * Essentially for every datum, new_datum = (datum - min_in_dataset) / (difference_dataset) * (difference_range) + range.first;
     * 
     * @param dataset Copy of the Dataset used to return the scaled dataset.
     * @return Dataset<Datum_t, Label_t> Scaled dataset.
     */
    virtual Dataset<Datum_t, Label_t> transform(Dataset<Datum_t, Label_t> dataset) const override;

    /**
     * @brief Scales the dataset according to the information obtained with the fit method, and only scales feature vectors
     * whose indices are contained in the fit_indices vector, outputting a new Dataset with those feature vectors, but scaled.
     * Essentially for every relevant datum, new_datum = (datum - min_in_dataset) / (difference_dataset) * (difference_range) + range.first;
     * 
     * @param dataset Constant reference to the Dataset that contains the feature vectors being scaled.
     * @param fit_indices Vector with the indices of the feature vectors that are going to be scaled.
     * @return Dataset<Datum_t, Label_t> New dataset with the chosen feature vectors scaled.
     */
    virtual Dataset<Datum_t, Label_t> transform(const Dataset<Datum_t, Label_t> &dataset, const std::vector<int> &transform_indices) const override;

    /**
     * @brief Applies the transformation inverse to scaling, obtained with transform.
     * Essentially, new_datum = (datum - range.first) / (difference_range) * (difference_dataset) + min_in_dataset;
     * 
     * @param datum Scaled feature vector meant to be returned to the original scaled.
     * @return Datum_t Feature vector at the original scale.
     */
    virtual Datum_t inverse_transform(Datum_t datum) const override;

    /**
     * @brief Gets the range.
     * 
     * @return std::pair<int, int> Range.
     */
    std::pair<int, int> get_range( ) const { return range; }

    /**
     * @brief Gets the feature-wise dataset minimum obtained through the fit method.
     * 
     * @return Datum_t Feature-wise dataset minimum obtained through the fit method.
     */
                Datum_t   get_min( ) const { return min_in_dataset; }

    /**
     * @brief Gets the feature-wise dataset maximum obtained through the fit method.
     * 
     * @return Datum_t Feature-wise dataset maximum obtained through the fit method.
     */
                Datum_t   get_max( ) const { return max_in_dataset; }

  protected:
    std::pair<int, int> range;  /**< Range to which each feature gets scaled. */

    int difference_range;       /**< Amplitude of the range. range.second - range.first. */

    Datum_t min_in_dataset;     /**< Feature-wise dataset minimum obtained through the fit method. */

    Datum_t max_in_dataset;     /**< Feature-wise dataset maximum obtained through the fit method. */

    Datum_t difference_dataset; /**< Difference between the feature-wise maximum and minimum. */



};

} // namespace tudat_learn

#include "tudat-learn/processing/scalers/minmax_scaler.tpp"

#endif // TUDAT_LEARN_MINMAX_SCALER_HPP