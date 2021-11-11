/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_RBF_H
#define TUDAT_LEARN_RBF_H

#include <memory>

#include "dataset.h"
#include "types.h"
#include "estimators/regressor.h"

namespace tudat_learn
{

struct RBF {
  virtual double eval(const double radius);
  // virtual double eval(const vector_double input_vector);

  /**
   * @brief Evaluation using two vectors.
   * 
   * @param x input vector
   * @param c center point
   * @return double 
   */
  virtual double eval(const vector_double &x, const vector_double &c) = 0;

  /**
   * @brief Evaluating the derivative using two vectors.
   * 
   * @param x input vector
   * @param c center point
   * @return std::shared_ptr<vector_double> jacobian at point x
   */
  virtual std::shared_ptr<vector_double> eval_derivative(const vector_double &x, const vector_double &c) = 0;

  virtual double eval_second_derivative(const double radius) = 0;
};


/**
 * @brief CubicRBF implements a simple Cubic Radial Basis Function.
 * 
 */
struct CubicRBF : public RBF {

  /**
   * @brief Construct a new CubicRBF object.
   * 
   */
  CubicRBF() {}

  /**
   * @brief Evaluation using the radius.
   * 
   * @param radius: Euclidean norm of the radius vector.
   * @return double 
   */
  double eval(const double radius) override final;

  /**
   * @brief Evaluation using two vectors.
   * 
   * @param x input vector
   * @param c center point
   * @return double 
   */
  double eval(const vector_double &x, const vector_double &c) override final;



  std::shared_ptr<vector_double> eval_derivative(const vector_double &x, const vector_double &c) override final;


  virtual double eval_second_derivative(const double radius);
};

/**
 * @brief CubicRBF implements a simple Gaussian Radial Basis Function.
 * 
 */
struct GaussianRBF : public RBF {
  
  /**
   * @brief Construct a new GaussianRBF object
   * 
   * @param sigma eval(x) = e(-x^2 / 2 * sigma^2)
   */
  GaussianRBF(const double sigma)
  : sigma(sigma) {}

  /**
   * @brief Evaluation using the radius.
   * 
   * @param radius: Euclidean norm of the radius vector.
   * @return double 
   */
  virtual double eval(const double radius) override final;

  /**
   * @brief Evaluation using two vectors.
   * 
   * @param x input vector
   * @param c center point
   * @return double 
   */
  double eval(const vector_double &x, const vector_double &c) override final;

  std::shared_ptr<vector_double> eval_derivative(const vector_double &x, const vector_double &c) override final;

  private:
    const double sigma;
};
  

} // namespace tudat_learn


#endif // TUDAT_LEARN_RBF_H