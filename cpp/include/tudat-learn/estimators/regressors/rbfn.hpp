/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_RBFN_HPP
#define TUDAT_LEARN_RBFN_HPP

#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "tudat-learn/dataset.hpp"
#include "tudat-learn/types.hpp"
#include "tudat-learn/estimators/regressor.hpp"
#include "tudat-learn/estimators/regressors/rbf.hpp"

namespace tudat_learn
{

/**
 * @brief Implementation of Radial Basis Function Networks. \n
 * First proposed by Rolland L. Hardy, in Multiquadric Equations of Topography and Other Irregular Surfaces.
 * More comprehensive explanation by  Natasha Flyer et al., in On the role of polynomials in RBF-FD approximations: 
 * I. Interpolation and accuracy.
 * 
 * @tparam Datum_t The type of a single feature vector. Has to be an Eigen::Vector (meaning: column vector) of a floating-point type. The
 * floating-point type must be the same as Label_t's.
 * @tparam Label_t The type of a single label. Has to be an Eigen::Vector (meaning: column vector) of a floating-point type. The
 * * floating-point type must be the same as Datum_t's.
 */
template <typename Datum_t, typename Label_t>
class RBFN : public Regressor<Datum_t, Label_t> {
  
  public:
    /**
     * @brief Deleting the default constructor to make sure the object is created with settings.
     */
    RBFN() = delete;

    /**
     * @brief Constructor for the RBFN class. \n 
     * Sets the shared pointers to the Dataset and to the RBF.
     * 
     * @tparam Datum_tt Same as Datum_t
     * @tparam Label_tt Same as Label_t
     * @tparam std::enable_if_t< is_floating_point_eigen_vector<Datum_tt>::value &&
     * is_floating_point_eigen_vector<Label_tt>::value &&
     * std::is_same<typename Datum_tt::Scalar, typename Label_tt::Scalar>::value
     * > Enables the constructor if and only if Datum_t and Label_t are of a floating-point Eigen::Vector type, and if their
     * elements are of the same type (for instance, two vectors of doubles or two vectors of floats).
     * @param dataset_ptr Constant reference to the Dataset pointer.
     * @param rbf_ptr Constant reference to the RBF pointer.
     */
    template < typename Datum_tt = Datum_t, typename Label_tt = Label_t, 
               typename = std::enable_if_t< is_floating_point_eigen_vector<Datum_tt>::value &&
                                            is_floating_point_eigen_vector<Label_tt>::value &&
                                            std::is_same<typename Datum_tt::Scalar, typename Label_tt::Scalar>::value
                          > 
    >
    RBFN(
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
    virtual ~RBFN( ) { }

    /**
     * @brief Override of the Regressor's fit( ) virtual member function. \n
     * Changes the center_points to the feature vectors in the Dataset and computes the coefficients by solving the linear system
     * given by Equation (2) in On the role of polynomials in RBF-FD approximations: I. Interpolation and accuracy. By Natasha
     * Flyer et al.. \n 
     * 
     */
    virtual void fit( ) override;

    /**
     * @brief Similar to the fit() function, while instead fitting the RBFN to feature vectors in the dataset at specific
     * indices only, instead of using the whole dataset.
     * 
     * @param fit_indices 
     */
    virtual void fit(const std::vector<size_t> &fit_indices) override;

    /**
     * @brief Override of the Regressor's eval() virtual member function. \n
     * Essentially implements, through matrix multiplications, Equation (1) in Natasha Flyer et al., On the role of polynomials
     * in RBF-FD approximations: I. Interpolation and accuracy.
     * 
     * @param input Constant reference to the point at which the RBFN value will be computed.
     * @return Label_t Predicted label, output of the RBFN at point input.
     */
    virtual Label_t eval(const Datum_t &input) const;

    /**
     * @brief Implementation of eval that takes advantege of Eigen's vectorization capabilities by computing predicted labels
     * for a vector of inputs.
     * 
     * @param input_vector Constant reference to vector of points at which the RBFN values will be computed.
     * @return Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> Predicted labels, outputs of the RBFN
     * at each of the input points. Each output corresponds to a row in the matrix.
     */
    virtual Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> eval(const std::vector<Datum_t> &input_vector) const;

    /**
     * @brief Returns the coefficients matrix.
     * 
     * @return const Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> coefficients
     */
    const Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> get_coefficients( ) const { return coefficients; }

    /**
     * @brief Computes the gradient of the RBFN at a point x.
     * 
     * @param x Point at which the gradient is computed.
     * @return Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> Gradient. 
     * Dimensions: #(output dimensions) by #(input dimensions). 
     */
    virtual             Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> gradient(const Datum_t &x) const;

    /**
     * @brief Computes the Hessians of the RBFN at a point x. \n 
     * 
     * @param x Point at which the Hessians are computed.
     * @return std::vector< Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> > Hessians.
     * Vector of size #(output dimensions), that is, one Hessian per output dimension, where each of them is a matrix of
     * #(input dimensions) by #(input dimensions).
     */
    virtual std::vector< Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> > hessians(const Datum_t &x) const;

  protected:
    std::shared_ptr< RBF<typename Datum_t::Scalar> > rbf_ptr; /**< Shared pointer to the RBF. */

    /**
     * @brief Matrix with the center points to which the RBFN is fitted.
     * Each center point is a row of the center_points matrix.
     * 
     */
    Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic > center_points;

    /**
     * @brief Matrix with the coefficients resulting from solving the linear system in Equation (2) in Natasha Flyer et al.,
     * On the role of polynomials in RBF-FD approximations: I. Interpolation and accuracy. Each row of the coefficients 
     * corresponds to a specific center point, while each of its columns corresponds to a specific output dimension. \n 
     * Has dimensions: #(center points) by #(output dimensions). \n 
     * When inherited by RBFNPolynomial, coefficients also has the polynomial coefficients, meaning its dimensions are
     * (#(center points) + 1 + #(input dimensions)) by #(output dimensions)
     * 
     */
    Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> coefficients; 
};
  
/**
 * @brief Implementation of Radial Basis Function Networks with first order polynomial terms.
 * Explanation in Natasha Flyer et al., On the role of polynomials in RBF-FD approximations: I. Interpolation and accuracy.
 * 
 * @tparam Datum_t The type of a single feature vector. Has to be an Eigen::Vector (meaning: column vector) of a floating-point type. The
 * floating-point type must be the same as Label_t's.
 * @tparam Label_t The type of a single label. Has to be an Eigen::Vector (meaning: column vector) of a floating-point type. The
 * * floating-point type must be the same as Datum_t's.
 */
template <typename Datum_t, typename Label_t>
class RBFNPolynomial : public RBFN<Datum_t, Label_t> {
  public:

    /**
     * @brief Deleting the default constructor to make sure the object is created with settings.
     */
    RBFNPolynomial() = delete;

    /**
     * @brief Constructor for the RBFNPolynomial class. \n
     * Sets the shared pointers to the Dataset and to the RBF. \n 
     * Like the RBFN it is derived from, the RBFNPolynomial constructor is only enabled if and only if Datum_t and Label_t
     * are of a floating-point Eigen::Vector type, and if their elements are of the same type (for instance, two vectors of 
     * doubles or two vectors of floats).
     * 
     * @param dataset_ptr Constant reference to the Dataset pointer.
     * @param rbf_ptr Constant reference to the RBF pointer.
     */
    RBFNPolynomial(
      const std::shared_ptr< Dataset<Datum_t, Label_t> > &dataset_ptr,
      const std::shared_ptr< RBF<typename Label_t::Scalar> > &rbf_ptr
    ) : 
    RBFN<Datum_t, Label_t>(dataset_ptr, rbf_ptr)
    { }

    /**
     * @brief Virtual destructor, as the class has virtual methods.
     * 
     */
    virtual ~RBFNPolynomial( ) { }

    /**
     * @brief Override of the Regressor's fit( ) virtual member function. \n
     * Changes the center_points to the feature vectors in the Dataset and computes the coefficients by solving the linear system
     * given by Equation (4) in On the role of polynomials in RBF-FD approximations: I. Interpolation and accuracy. By Natasha
     * Flyer et al.. \n 
     * 
     */
    virtual void fit( ) override;

    /**
     * @brief Similar to the fit() function, while instead fitting the RBFNPolynomial to feature vectors in the dataset at specific
     * indices only, instead of using the whole dataset.
     * 
     * @param fit_indices Vector with the indices of the feature vectors to which the RBFNPolynomial is going to be fitted.
     */
    virtual void fit(const std::vector<size_t> &fit_indices) override;

    /**
     * @brief Override of the Regressor's eval() virtual member function. \n
     * Essentially implements, through matrix multiplications, Equation (3)  in Natasha Flyer et al., On the role of polynomials
     * in RBF-FD approximations: I. Interpolation and accuracy.
     * 
     * @param input Constant reference to the point at which the RBFN value will be computed.
     * @return Label_t Predicted label, output of the RBFN at point input.
     */
    virtual Label_t eval(const Datum_t &input) const override;

    /**
     * @brief Impelementation of eval() that takes advantege of Eigen's vectorization capabilities by computing predicted labels
     * for a vector of inputs.
     * 
     * @param input_vector Constant reference to vector of points at which the RBFNPolynomial values will be computed
     * @return Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> Predicted labels, outputs of the
     * RBFNPolynomial at each of the input points. EAch output corresponds to a row in the matrix.
     */
    virtual Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> eval(const std::vector<Datum_t> &input_vector) const;

    /**
     * @brief Computes the gradient of the RBFNPolynomial at a point x.
     * 
     * @param x Point at which the gradient is computed.
     * @return Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> Gradient. 
     * Dimensions: #(output dimensions) by #(input dimensions). 
     */
    virtual Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> gradient(const Datum_t &x) const override;

    /**
     * @brief Computes the Hessians of the RBFNPolynomial at a point x. \n 
     * 
     * @param x Point at which the Hessians are computed.
     * @return std::vector< Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> > Hessians.
     * Vector of size #(output dimensions), that is, one Hessian per output dimension, where each of them is a matrix of
     * #(input dimensions) by #(input dimensions).
     */
    virtual std::vector< Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> > hessians(const Datum_t &x) const override;

};

} // namespace tudat_learn

#include "tudat-learn/estimators/regressors/rbfn.tpp"

#endif // TUDAT_LEARN_RBFN_HPP
