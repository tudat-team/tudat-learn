/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_RBF_HPP
#define TUDAT_LEARN_RBF_HPP

#include <memory>
#include <type_traits>
#include <vector>

#include <Eigen/Core>

#include "tudat-learn/dataset.hpp"
#include "tudat-learn/types.hpp"
#include "tudat-learn/estimators/regressor.hpp"

namespace tudat_learn
{

template <typename T>
struct RBF {

  /**
     * @brief Virtual destructor, as the class has virtual methods.
     * 
     */
  virtual ~RBF( ) { }

  /**
   * @brief Types used in the RBFs
   * 
   * @param vector_t STL vector of type T 
   * @param VectorXt Eigen::Vector of type T, Dynamic number of rows, single column
   * @param MatrixXt Eigen::Matrix of type T, Dynamic number of rows amd columns
   * 
   */
  using vector_t = std::vector< T >; /**<  STL vector of type T. */
  using VectorXt = Eigen::Matrix< T, Eigen::Dynamic, 1 >; /**< Eigen::Vector of type T, Dynamic number of rows, single column. */
  using MatrixXt = Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >; /**< Eigen::Matrix of type T, Dynamic number of rows amd columns. */

  /**
   * @brief Evaluation using the radius.
   * 
   * @param radius: Euclidean norm of the radius vector.
   * @return T 
   */
  virtual T eval(const T radius) const = 0;

  /**
   * @brief Applies the RBF to every value of a matrix.
   * 
   * @param distance_matrix Receives a matrix with euclidean distances between the input and a number of center points
   * @return MatrixXt output matrix
   */
  virtual MatrixXt eval_matrix(const MatrixXt &distance_matrix) const = 0;

  /**
   * @brief Evaluation using two std::vector<T>.
   * 
   * @param x input vector
   * @param c center point
   * @return T
   */
  virtual T eval(const vector_t &x, const vector_t &c) const = 0;

  /**
   * @brief Evaluation using two VectorXt.
   * 
   * @param x input vector
   * @param c center point
   * @return T 
   */
  virtual T eval(const VectorXt &x, const VectorXt &c) const = 0;
  
  /**
   * @brief Evaluating the gradient using two std::vector<T>.
   * 
   * @param x input vector
   * @param c center point
   * @return std::shared_ptr<vector_t> gradient at point x
   */
  virtual std::shared_ptr<vector_t> eval_gradient(const vector_t &x, const vector_t &c) const = 0;

  /**
   * @brief Evaluating the gradient using two VectorXt.
   * 
   * @param x input vector
   * @param c center point
   * @return std::shared_ptr<vector_t> gradient at point x
   */
  virtual std::shared_ptr<VectorXt> eval_gradient(const VectorXt &x, const VectorXt &c) const = 0;

  /**
   * @brief Evaluating the hessian using two std::vector<T>.
   * 
   * @param x input vector
   * @param c center point
   * @return std::shared_ptr<vector_t> hessian with indexing j * x.size() + i yields (d^2 f) / (dxi dxj) derivative
   * @return std::shared_ptr<MatrixXt> hessian with indexing (i,j) yields (d^2 f) / (dxi dxj) derivative
   */
  virtual std::shared_ptr<vector_t> eval_hessian(const vector_t &x, const vector_t &c) const = 0;

  /**
   * @brief Evaluating the hessian using two VectorXt.
   * 
   * @param x input vector
   * @param c center point
   * @return std::shared_ptr<vector_t> hessian with indexing j * x.size() + i yields (d^2 f) / (dxi dxj) derivative
   * @return std::shared_ptr<MatrixXt> hessian with indexing (i,j) yields (d^2 f) / (dxi dxj) derivative
   */
  virtual std::shared_ptr<MatrixXt> eval_hessian(const VectorXt &x, const VectorXt &c) const = 0;
  
  
  /**
   * @brief Computes a matrix of RBF partial derivatives so that the gradient of the RBFN at the point x
   * can be computed using:
   * coefficients.transpose() * gradient_rbfn(x, center_points)
   * 
   * In case of an RBFNPolynomial, one just needs to concatenate a matrix of dimensions 
   * [(dimension_input + 1) x dimension_output] below the output of this function before the multiplication.
   * The matrix shall have the form presented below:
   * 0 0 ... 0
   * 1 1 ... 1
   * 1 1 ... 1
   *   ...
   * 1 1 ... 1
   * 
   * Done to take advantage of Eigen's vectorization capabilities.
   * 
   * @param x point at which the gradient is to be computed
   * @param center_points center points with which the RBFN was built. Matrix should be of size 
   * [(#center points) x dimension_input], with the center point vectors being displayed horizontally.
   * @return MatrixXt [(#center points) x dimension_input] matrix that it is multiplied by the 
   * [(#center points) x dimension input].transposed() coefficient matrix yields the 
   * [dimension_output x dimension_input] gradient.
   */
  virtual             MatrixXt  gradient_rbfn(const VectorXt &x, const MatrixXt &center_points) const = 0;

  /**
   * @brief Computes a vector of [(#center points) x dimension_input] matrices with part of the second
   * partial derivatives of the RBFN or RBFN polynomial functions. In order to obtain the Hessian of the output 
   * dimension k, one should transpose and multiply the RBFN coefficients of the k-th dimension, that is,
   * RBFN::coefficients.col(k).transpose(), by each of the matrices in the vector produced by this function.
   * Each multiplication will yield a row of the corresponding Hessian. 
   * 
   * This multiplication process should be repeated for each set of coefficients (each output dimension)
   * in order to obtain the Hessian matrices for every output dimension.
   * 
   * Done to take advantage of Eigen's vectorization capabilities.
   * 
   * @param x point at which the gradient is to be computed
   * @param center_points center points with which the RBFN was built. Matrix should be of size 
   * [(#center points) x dimension_input], with the center point vectors being displayed horizontally.
   * @return std::vector<MatrixXt> vector of [(#center points) x dimension_input] matrices that when multiplied
   * by the coefficient vectors yield the Hessian matrix associated with the respective output dimensions.
   */
  virtual std::vector<MatrixXt>  hessian_rbfn(const VectorXt &x, const MatrixXt &center_points) const = 0;

  
};


/**
 * @brief CubicRBF implements a simple Cubic Radial Basis Function.
 * 
 */
template <typename T>
struct CubicRBF : public RBF<T> {
  using typename RBF<T>::vector_t;
  using typename RBF<T>::VectorXt;
  using typename RBF<T>::MatrixXt;

  /**
   * @brief Construct a new CubicRBF object.
   * 
   */
  CubicRBF() : RBF< T >() {}

  /**
     * @brief Virtual destructor, as the class has virtual methods.
     * 
     */
  virtual ~CubicRBF( ) { }

  /**
   * @brief Evaluation using the radius.
   * 
   * @param radius: Euclidean norm of the radius vector.
   * @return T Output of the RBF 
   */
  virtual T eval(const T radius) const override final;

  /**
   * @brief Cubes (^3) every value of a matrix.
   * 
   * @param distance_matrix Receives a matrix with euclidean distances between the input and a number of center points
   * @return MatrixXt output matrix
   */
  virtual MatrixXt eval_matrix(const MatrixXt &distance_matrix) const override final;
  
  /**
   * @brief Evaluation using two std::vector<T>.
   * 
   * @param x input vector
   * @param c center point
   * @return T Output of the RBF
   */
  virtual T eval(const vector_t &x, const vector_t &c) const override final;

  /**
   * @brief Evaluation using two VectorXt.
   * 
   * @param x input vector
   * @param c center point
   * @return T Output of the RBF
   */
  virtual T eval(const VectorXt &x, const VectorXt &c) const override final;
  
  /**
   * @brief Evaluating the gradient using two std::vector<T>.
   * 
   * @param x input vector
   * @param c center point
   * @return std::shared_ptr<vector_t> gradient at point x
   */
  virtual std::shared_ptr<vector_t> eval_gradient(const vector_t &x, const vector_t &c) const override final;

  /**
   * @brief Evaluating the gradient using two VectorXt.
   * 
   * @param x input vector
   * @param c center point
   * @return std::shared_ptr<vector_t> gradient at point x
   */
  virtual std::shared_ptr<VectorXt> eval_gradient(const VectorXt &x, const VectorXt &c) const override final;

  /**
   * @brief Evaluating the hessian using two std::vector<T>.
   * 
   * @param x input vector
   * @param c center point
   * @return std::shared_ptr<vector_t> hessian with indexing j * x.size() + i yields (d^2 f) / (dxi dxj) derivative
   * @return std::shared_ptr<MatrixXt> hessian with indexing (i,j) yields (d^2 f) / (dxi dxj) derivative
   */
  virtual std::shared_ptr<vector_t> eval_hessian(const vector_t &x, const vector_t &c) const override final;

  /**
   * @brief Evaluating the hessian using two VectorXt.
   * 
   * @param x input vector
   * @param c center point
   * @return std::shared_ptr<vector_t> hessian with indexing j * x.size() + i yields (d^2 f) / (dxi dxj) derivative
   * @return std::shared_ptr<MatrixXt> hessian with indexing (i,j) yields (d^2 f) / (dxi dxj) derivative
   */
  virtual std::shared_ptr<MatrixXt> eval_hessian(const VectorXt &x, const VectorXt &c) const override final;

  /**
   * @brief Computes a matrix of RBF partial derivatives so that the gradient of the RBFN at the point x
   * can be computed using:
   * coefficients.transpose() * gradient_rbfn(x, center_points)
   * 
   * In case of an RBFNPolynomial, one just needs to concatenate a matrix of dimensions 
   * [(dimension_input + 1) x dimension_output] below the output of this function before the multiplication.
   * The matrix shall have the form presented below:
   * 0 0 ... 0
   * 1 1 ... 1
   * 1 1 ... 1
   *   ...
   * 1 1 ... 1
   * 
   * Done to take advantage of Eigen's vectorization capabilities.
   * 
   * @param x point at which the gradient is to be computed
   * @param center_points center points with which the RBFN was built. Matrix should be of size 
   * [(#center points) x dimension_input], with the center point vectors being displayed horizontally.
   * @return MatrixXt [(#center points) x dimension_input] matrix that it is multiplied by the 
   * [(#center points) x dimension input].transposed() coefficient matrix yields the 
   * [dimension_output x dimension_input] gradient.
   */
  virtual             MatrixXt  gradient_rbfn(const VectorXt &x, const MatrixXt &center_points) const override final;

  /**
   * @brief Computes a vector of [(#center points) x dimension_input] matrices with part of the second
   * partial derivatives of the RBFN or RBFN polynomial functions. In order to obtain the Hessian of the output 
   * dimension k, one should transpose and multiply the RBFN coefficients of the k-th dimension, that is,
   * RBFN::coefficients.col(k).transpose(), by each of the matrices in the vector produced by this function.
   * Each multiplication will yield a row of the corresponding Hessian. 
   * 
   * This multiplication process should be repeated for each set of coefficients (each output dimension)
   * in order to obtain the Hessian matrices for every output dimension.
   * 
   * Done to take advantage of Eigen's vectorization capabilities.
   * 
   * @param x point at which the gradient is to be computed
   * @param center_points center points with which the RBFN was built. Matrix should be of size 
   * [(#center points) x dimension_input], with the center point vectors being displayed horizontally.
   * @return std::vector<MatrixXt> vector of [(#center points) x dimension_input] matrices that when multiplied
   * by the coefficient vectors yield the Hessian matrix associated with the respective output dimensions.
   */
  virtual std::vector<MatrixXt>  hessian_rbfn(const VectorXt &x, const MatrixXt &center_points) const override final;
};

/**
 * @brief CubicRBF implements a simple Gaussian Radial Basis Function.
 * 
 */
template <typename T>
struct GaussianRBF : public RBF<T> {
  using typename RBF<T>::vector_t;
  using typename RBF<T>::VectorXt;
  using typename RBF<T>::MatrixXt;
  
  /**
   * @brief Deleted default constructor, ensuring the objects are constructed with settings.
   * 
   */
  GaussianRBF() = delete;

  /**
   * @brief Construct a new GaussianRBF object
   * 
   * @param sigma eval(x) = e(-x^2 / 2 * sigma^2)
   */
  GaussianRBF(const double sigma)
  : RBF< T >(), sigma_sqrd(sigma * sigma) {}

  /**
     * @brief Virtual destructor, as the class has virtual methods.
     * 
     */
  virtual ~GaussianRBF( ) { }

  /**
   * @brief Evaluation using the radius.
   * 
   * @param radius: Euclidean norm of the radius vector.
   * @return T Output of the RBF
   */
  virtual T eval(const T radius) const override final;

  /**
   * @brief Applies the Gaussian RBF to every value of a matrix.
   * 
   * @param distance_matrix Receives a matrix with euclidean distances between the input and a number of center points
   * @return MatrixXt output matrix
   */
  virtual MatrixXt eval_matrix(const MatrixXt &distance_matrix) const override final;

  /**
   * @brief Evaluation using two std::vector<T>.
   * 
   * @param x input vector
   * @param c center point
   * @return T Output of the RBF
   */
  virtual T eval(const vector_t &x, const vector_t &c) const override final;

  /**
   * @brief Evaluation using two VectorXt.
   * 
   * @param x input vector
   * @param c center point
   * @return T Output of the RBF
   */
  virtual T eval(const VectorXt &x, const VectorXt &c) const override final;
  
  /**
   * @brief Evaluating the gradient using two std::vector<T>.
   * 
   * @param x input vector
   * @param c center point
   * @return std::shared_ptr<vector_t> gradient at point x
   */
  virtual std::shared_ptr<vector_t> eval_gradient(const vector_t &x, const vector_t &c) const override final;

  /**
   * @brief Evaluating the gradient using two VectorXt.
   * 
   * @param x input vector
   * @param c center point
   * @return std::shared_ptr<vector_t> gradient at point x
   */
  virtual std::shared_ptr<VectorXt> eval_gradient(const VectorXt &x, const VectorXt &c) const override final;

  /**
   * @brief Evaluating the hessian using two std::vector<T>.
   * 
   * @param x input vector
   * @param c center point
   * @return std::shared_ptr<vector_t> hessian with indexing j * x.size() + i yields (d^2 f) / (dxi dxj) derivative
   * @return std::shared_ptr<MatrixXt> hessian with indexing (i,j) yields (d^2 f) / (dxi dxj) derivative
   */
  virtual std::shared_ptr<vector_t> eval_hessian(const vector_t &x, const vector_t &c) const override final;

  /**
   * @brief Evaluating the hessian using two VectorXt.
   * 
   * @param x input vector
   * @param c center point
   * @return std::shared_ptr<vector_t> hessian with indexing j * x.size() + i yields (d^2 f) / (dxi dxj) derivative
   * @return std::shared_ptr<MatrixXt> hessian with indexing (i,j) yields (d^2 f) / (dxi dxj) derivative
   */
  virtual std::shared_ptr<MatrixXt> eval_hessian(const VectorXt &x, const VectorXt &c) const override final;


  /**
   * @brief Computes a matrix of RBF partial derivatives so that the gradient of the RBFN at the point x
   * can be computed using:
   * coefficients.transpose() * gradient_rbfn(x, center_points) \n
   * 
   * In case of an RBFNPolynomial, one just needs to concatenate a matrix of dimensions 
   * [(dimension_input + 1) x dimension_input] below the output of this function before the multiplication.
   * The matrix shall have the form presented below: \n
   * 0 0 ... 0 \n
   * 1 1 ... 1 \n
   * 1 1 ... 1 \n
   *   ...     \n
   * 1 1 ... 1 \n
   * 
   * Done to take advantage of Eigen's vectorization capabilities.
   * 
   * @param x point at which the gradient is to be computed
   * @param center_points center points with which the RBFN was built. Matrix should be of size 
   * [(#center points) x dimension_input], with the center point vectors being displayed horizontally.
   * @return MatrixXt [(#center points) x dimension_input] matrix that it is multiplied by the 
   * [(#center points) x dimension input].transposed() coefficient matrix yields the 
   * [dimension_output x dimension_input] gradient.
   */
  virtual             MatrixXt  gradient_rbfn(const VectorXt &x, const MatrixXt &center_points) const override final;

  /**
   * @brief Computes a vector of [(#center points) x dimension_input] matrices with part of the second order
   * partial derivatives of the RBFN or RBFN polynomial functions. In order to obtain the Hessian of the output 
   * dimension k, one should transpose and multiply the RBFN coefficients of the k-th dimension, that is,
   * RBFN::coefficients.col(k).transpose(), by each of the matrices in the vector produced by this function.
   * Each multiplication will yield a row of the corresponding Hessian. 
   * 
   * This multiplication process should be repeated for each set of coefficients (each output dimension)
   * in order to obtain the Hessian matrices for every output dimension.
   * 
   * Done to take advantage of Eigen's vectorization capabilities.
   * 
   * @param x point at which the gradient is to be computed
   * @param center_points center points with which the RBFN was built. Matrix should be of size 
   * [(#center points) x dimension_input], with the center point vectors being displayed horizontally.
   * @return std::vector<MatrixXt> vector of [(#center points) x dimension_input] matrices that when multiplied
   * by the coefficient vectors yield the Hessian matrix associated with the respective output dimensions.
   */
  virtual std::vector<MatrixXt>  hessian_rbfn(const VectorXt &x, const MatrixXt &center_points) const override final;


  protected:
    const T sigma_sqrd; /**< Square of the sigma value provided as input. */
};
  
} // namespace tudat_learn


#include "tudat-learn/estimators/regressors/rbf.tpp"

#endif // TUDAT_LEARN_RBF_HPP
