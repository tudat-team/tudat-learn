/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_GRNN_HPP
#define TUDAT_LEARN_GRNN_HPP

#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <Eigen/Core>

#include "tudat-learn/dataset.hpp"
#include "tudat-learn/types.hpp"
#include "tudat-learn/estimators/regressor.hpp"
#include "tudat-learn/estimators/regressors/rbf.hpp"

namespace tudat_learn
{

/**
 * @brief Implementation of Generalized Regression Neural Networks.
 * From: D. F. Specht, A general regression neural network.
 * 
 * @tparam Datum_t The type of a single feature vector. Has to be an Eigen::Vector (meaning: column vector) of a floating-point type. The
 * floating-point type must be the same as Label_t's.
 * @tparam Label_t The type of a single label. Has to be an Eigen::Vector (meaning: column vector) of a floating-point type. The
 * * floating-point type must be the same as Datum_t's.
 */
template <typename Datum_t, typename Label_t>
class GRNN : public Regressor<Datum_t, Label_t> {

  public:
    /**
    * @brief Deleting the default constructor to make sure the object is created with settings.
    */
    GRNN() = delete;

    /**
     * @brief Constructor for the GRNN class. \n
     * Sets the shared pointers to the Dataset and to the RBF.
     * 
     * 
     * @tparam Datum_tt Same as Datum_t.
     * @tparam Label_tt Same as Label_t.
     * @tparam std::enable_if_t< is_floating_point_eigen_vector<Datum_tt>::value &&
     * is_floating_point_eigen_vector<Label_tt>::value &&
     * std::is_same<typename Datum_tt::Scalar, typename Label_tt::Scalar>::value
     * > Enables the constructor if and only if Datum_t and Label_t are of a floating-point Eigen::Vector type, and if their 
     * elements are of the same type (for instance, two vectors of doubles or two vectors of floats).
     * @param dataset_ptr Constant reference to the Dataset pointer.
     * @param rbf_ptr Constant reference to the RBF pointer.
     */
    template <typename Datum_tt = Datum_t, typename Label_tt = Label_t,
              typename = std::enable_if_t< is_floating_point_eigen_vector<Datum_tt>::value &&
                                           is_floating_point_eigen_vector<Label_tt>::value &&
                                           std::is_same<typename Datum_tt::Scalar, typename Label_tt::Scalar>::value
                          >
    >
    GRNN(
      const std::shared_ptr< Dataset<Datum_tt, Label_tt> > &dataset_ptr,
      const std::shared_ptr< RBF<typename Datum_tt::Scalar> > &rbf_ptr
    ) :
    Regressor<Datum_tt, Label_tt>(dataset_ptr),
    rbf_ptr(rbf_ptr)
    { }

    /**
     * @brief Virtual destructor, as the class has virtual methods.
     * 
     */
    virtual ~GRNN( ) { }

    /**
     * @brief Override of the Regressor's fit( ) virtual member function. \n
     * Changes the center_points and output_center_points variables. Ready to eval( ) afterwards. 
     * 
     */
    virtual void fit( ) override;

    /**
     * @brief Similar to the fit() function, while instead fitting the GRNN to feature vectors in the dataset at specific
     * indices only, instead of using the whole dataset.
     * 
     * @param fit_indices Vector with the indices of the feature vectors to which the GRNN is going to be fitted.
     */
    virtual void fit(const std::vector<size_t> &fit_indices) override;

    /**
     * @brief Override of the Regressor's eval() virtual member function. \n
     * Essentially implements, through matrix multiplications, Equation (5) in D. F. Specht, A general regression neural network.
     * 
     * @param input Constant reference to the point at which the GRNN value will be computed.
     * @return Label_t Predicted label, output of the GRNN at point input.
     */
    virtual Label_t eval(const Datum_t &input) const override;

    /**
     * @brief Implementation of eval() that takes advantage of Eigen's vectorization capabilities by computing predicted labels
     * for a vector of inputs.
     * 
     * @param input_vector Constant reference to vector of points at which the GRNN values will be computed.
     * @return Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> Predicted labels, outputs of the GRNN
     * at each of the input points. Each output corresponds to a row in the matrix.
     */
    virtual Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> eval(const std::vector<Datum_t> &input_vector) const;    

  protected:
    std::shared_ptr< RBF<typename Datum_t::Scalar> > rbf_ptr; /**< Shared pointer to the RBF. */

    /**
     * @brief Matrix with the center points to which the GRNN is fitted.
     * Each center point is a row of the center_points matrix.
     * 
     */
    Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic > center_points;

    /**
     * @brief Matrix with the labels of the center points to which the GRNN is fitted.
     * Each label is a row of the output_center_points matrix.
     * 
     */
    Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> output_center_points; 
};

} // namespace tudat_learn

#include "tudat-learn/estimators/regressors/grnn.tpp"

#endif // TUDAT_LEARN_GRNN_HPP