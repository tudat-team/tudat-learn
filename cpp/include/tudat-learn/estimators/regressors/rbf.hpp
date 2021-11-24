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

#include <Eigen/Core>

#include "tudat-learn/dataset.hpp"
#include "tudat-learn/types.hpp"
#include "tudat-learn/estimators/regressor.hpp"

namespace tudat_learn
{

template <typename T>
struct RBF {
  using vector_t = std::vector< T >;
  using VectorXt = Eigen::Matrix< T, Eigen::Dynamic, 1 >;
  using MatrixXt = Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic >;

  virtual T eval(const T radius) const = 0;
  // virtual double eval(const vector_t input_vector);

  /**
   * @brief Evaluation using two vectors.
   * 
   * @param x input vector
   * @param c center point
   * @return double 
   */
  virtual T eval(const vector_t &x, const vector_t &c) const = 0;
  virtual T eval(const VectorXt &x, const VectorXt &c) const = 0;

  /**
   * @brief Evaluating the jacobian using two vectors.
   * 
   * @param x input vector
   * @param c center point
   * @return std::shared_ptr<vector_t> jacobian at point x
   */
  virtual std::shared_ptr<vector_t> eval_jacobian(const vector_t &x, const vector_t &c) const = 0;
  virtual std::shared_ptr<VectorXt> eval_jacobian(const VectorXt &x, const VectorXt &c) const = 0;

  /**
   * @brief Evaluating the hessian using two vectors
   * 
   * @param x input vector
   * @param c center point
   * @return std::shared_ptr<vector_t> hessian with indexing j * x.size() + i yields (d^2 f) / (dxi dxj) derivative
   */
  virtual std::shared_ptr<vector_t> eval_hessian(const vector_t &x, const vector_t &c) const = 0;
  virtual std::shared_ptr<MatrixXt> eval_hessian(const VectorXt &x, const VectorXt &c) const = 0;
  
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
   * @brief Evaluation using the radius.
   * 
   * @param radius: Euclidean norm of the radius vector.
   * @return double 
   */
  T eval(const T radius) const override final;
  
  /**
   * @brief Evaluation using two vectors.
   * 
   * @param x input vector
   * @param c center point
   * @return double 
   */
  T eval(const vector_t &x, const vector_t &c) const override final;
  T eval(const VectorXt &x, const VectorXt &c) const override final;
  
  /**
   * @brief Evaluating the jacobian using two vectors.
   * 
   * @param x input vector
   * @param c center point
   * @return std::shared_ptr<vector_t> jacobian at point x
   */
  std::shared_ptr<vector_t> eval_jacobian(const vector_t &x, const vector_t &c) const override final;
  std::shared_ptr<VectorXt> eval_jacobian(const VectorXt &x, const VectorXt &c) const override final;

  /**
   * @brief Evaluating the hessian using two vectors
   * 
   * @param x input vector
   * @param c center point
   * @return std::shared_ptr<vector_t> hessian with indexing j * x.size() + i yields (d^2 f) / (dxi dxj) derivative
   */
  std::shared_ptr<vector_t> eval_hessian(const vector_t &x, const vector_t &c) const override final;
  std::shared_ptr<MatrixXt> eval_hessian(const VectorXt &x, const VectorXt &c) const override final;
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
   * @brief Construct a new GaussianRBF object
   * 
   * @param sigma eval(x) = e(-x^2 / 2 * sigma^2)
   */
  GaussianRBF(const double sigma)
  : RBF< T >(), sigma_sqrd(sigma * sigma) {}

  /**
   * @brief Evaluation using the radius.
   * 
   * @param radius: Euclidean norm of the radius vector.
   * @return double 
   */
  T eval(const T radius) const override final;

  /**
   * @brief Evaluation using two vectors.
   * 
   * @param x input vector
   * @param c center point
   * @return double 
   */
  T eval(const vector_t &x, const vector_t &c) const override final;
  T eval(const VectorXt &x, const VectorXt &c) const override final;
  
  /**
   * @brief Evaluating the jacobian using two vectors.
   * 
   * @param x input vector
   * @param c center point
   * @return std::shared_ptr<vector_t> jacobian at point x
   */
  std::shared_ptr<vector_t> eval_jacobian(const vector_t &x, const vector_t &c) const override final;
  std::shared_ptr<VectorXt> eval_jacobian(const VectorXt &x, const VectorXt &c) const override final;

  /**
   * @brief Evaluating the hessian using two vectors
   * 
   * @param x input vector
   * @param c center point
   * @return std::shared_ptr<vector_t> hessian with indexing j * x.size() + i yields (d^2 f) / (dxi dxj) derivative
   * @return std::shared_ptr<MatrixXt> hessian with indexing (i,j) yields (d^2 f) / (dxi dxj) derivative
   */
  std::shared_ptr<vector_t> eval_hessian(const vector_t &x, const vector_t &c) const override final;
  std::shared_ptr<MatrixXt> eval_hessian(const VectorXt &x, const VectorXt &c) const override final;


  private:
    const T sigma_sqrd;
};
  
} // namespace tudat_learn


#include "tudat-learn/estimators/regressors/rbf.tpp"

#endif // TUDAT_LEARN_RBF_HPP
