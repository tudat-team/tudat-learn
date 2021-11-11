/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#include <cmath>
#include <memory>
#include <stdexcept>

#include "types.h"
#include "estimators/regressors/rbf.h"

namespace tudat_learn
{

double CubicRBF::eval(const double radius) {
  return radius * radius * radius;
}

double CubicRBF::eval(const vector_double &x, const vector_double &c) {
  if( x.size() != c.size())
    throw std::runtime_error("Vector dimensions are not the same in  CubicRBF::eval(const vector_double x, const vector_double c)");

  double result = 0;

  for(auto j = 0u; j < x.size(); ++j)
    result += (x[j] - c[j]) * (x[j] - c[j]);

  result = std::sqrt(result);
  result = result * result * result;

  return result;
}

std::shared_ptr<vector_double> CubicRBF::eval_derivative(const vector_double &x, const vector_double &c) {
  if( x.size() != c.size())
    throw std::runtime_error("Vector dimensions are not the same in  CubicRBF::eval_derivative(const vector_double x, const vector_double c)");
  
  auto jacobian_at_x = std::make_shared<vector_double>();

  jacobian_at_x->reserve(x.size());
  
  double radius = 0;

  for(auto j = 0u; j < x.size(); ++j)
    radius += (x[j] - c[j]) * (x[j] - c[j]);

  radius = std::sqrt(radius);

  for(auto j = 0u; j < x.size(); ++j)
    jacobian_at_x.get()->at(j) = 3 * (x[j] - c[j]) * radius;

  return jacobian_at_x;
}

double GaussianRBF::eval(const double radius) {
  return std::exp(- (radius * radius) / (2 * sigma));
}

double GaussianRBF::eval(const vector_double &x, const vector_double &c) {
  if( x.size() != c.size())
    throw std::runtime_error("Vector dimensions are not the same in  GaussianRBF::eval(const vector_double x, const vector_double c)");

  double result = 0;

  for(auto j = 0u; j < x.size(); ++j)
    result += (x[j] - c[j]) * (x[j] - c[j]);

  result = std::exp( - result / (2 * sigma * sigma));

  return result;
}

std::shared_ptr<vector_double> GaussianRBF::eval_derivative(const vector_double &x, const vector_double &c) {
  if( x.size() != c.size())
    throw std::runtime_error("Vector dimensions are not the same in  GaussianRBF::eval_derivative(const vector_double x, const vector_double c)");
  
  auto jacobian_at_x = std::make_shared<vector_double>();

  jacobian_at_x->reserve(x.size());
  
  double radius = 0;
  double eval_at_x = 0;

  for(auto j = 0u; j < x.size(); ++j)
    radius += (x[j] - c[j]) * (x[j] - c[j]);

  // use squared radius for evaluation before the sqrt
  eval_at_x = std::exp( - radius / (2 * sigma));

  radius = std::sqrt(radius);

  for(auto j = 0u; j < x.size(); ++j)
    jacobian_at_x.get()->at(j) = eval_at_x * (- (x[j] - c[j]) / (radius * sigma * sigma));

  return jacobian_at_x;
}
  
} // namespace tudat_learn