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
#include <vector>
#include <stdexcept>

#include "types.h"
#include "estimators/regressors/rbf.h"

namespace tudat_learn
{

// CubicRBF //

double CubicRBF::eval(const double radius) {
  return radius * radius * radius;
}

double CubicRBF::eval(const vector_double &x, const vector_double &c) {
  if( x.size() != c.size())
    throw std::runtime_error("Vector dimensions are not the same in  CubicRBF::eval(const vector_double x, const vector_double c)");

  double result = 0;

  for(auto j = 0; j < x.size(); ++j)
    result += (x[j] - c[j]) * (x[j] - c[j]);

  result = std::sqrt(result);
  result = result * result * result; // faster than pow(result, 3/2)

  return result;
}

std::shared_ptr<vector_double> CubicRBF::eval_jacobian(const vector_double &x, const vector_double &c) {
  if( x.size() != c.size())
    throw std::runtime_error("Vector dimensions are not the same in  CubicRBF::eval_derivative(const vector_double x, const vector_double c)");
  
  auto jacobian_at_x = std::make_shared<vector_double>();

  jacobian_at_x->reserve(x.size());
  
  double radius = 0;

  for(auto j = 0; j < x.size(); ++j)
    radius += (x[j] - c[j]) * (x[j] - c[j]);

  radius = std::sqrt(radius);

  for(auto j = 0; j < x.size(); ++j)
    jacobian_at_x.get()->push_back(3 * (x[j] - c[j]) * radius);

  return jacobian_at_x;
}

std::shared_ptr< vector_double > CubicRBF::eval_hessian(const vector_double &x, const vector_double &c) {
  if( x.size() != c.size())
    throw std::runtime_error("Vector dimensions are not the same in  CubicRBF::eval_second_derivative(const vector_double x, const vector_double c)");
  
  auto hessian_at_x = std::make_shared< vector_double>(x.size() * x.size());

  double radius = 0;

  for(auto j = 0; j < x.size(); ++j)
    radius += (x[j] - c[j]) * (x[j] - c[j]);

  radius = std::sqrt(radius);

  for(auto j = 0; j < x.size(); ++j) {
    for(auto k = j; k < x.size(); ++k) {
      hessian_at_x.get()->at(j * x.size() + k) = 3 * (x[k] - c[k]) * (x[j] - c[j]) / radius;

      // adding term to the diagonal
      if(j == k)
        hessian_at_x.get()->at(j * x.size() + k) += 3 * radius;
      // copying the value to the transposed position if it is not part of the diagonal
      else
        hessian_at_x.get()->at(k * x.size() + j) = hessian_at_x.get()->at(j * x.size() + k);
    }
  }

  return hessian_at_x;
}

// GaussianRBF //

double GaussianRBF::eval(const double radius) {
  return std::exp(- (radius * radius) / (sigma_sqrd));
}

double GaussianRBF::eval(const vector_double &x, const vector_double &c) {
  if( x.size() != c.size())
    throw std::runtime_error("Vector dimensions are not the same in  GaussianRBF::eval(const vector_double x, const vector_double c)");

  double result = 0;

  for(auto j = 0; j < x.size(); ++j)
    result += (x[j] - c[j]) * (x[j] - c[j]);

  result = std::exp( - result / (sigma_sqrd));

  return result;
}

std::shared_ptr<vector_double> GaussianRBF::eval_jacobian(const vector_double &x, const vector_double &c) {
  if( x.size() != c.size())
    throw std::runtime_error("Vector dimensions are not the same in  GaussianRBF::eval_derivative(const vector_double x, const vector_double c)");
  
  auto jacobian_at_x = std::make_shared<vector_double>();

  jacobian_at_x->reserve(x.size());
  
  double gaussian_at_x = eval(x, c);

  for(auto j = 0; j < x.size(); ++j)
    jacobian_at_x.get()->push_back(gaussian_at_x * (-2 * (x[j] - c[j]) / (sigma_sqrd)));

  return jacobian_at_x;
}

std::shared_ptr< vector_double > GaussianRBF::eval_hessian(const vector_double &x, const vector_double &c) {
  if( x.size() != c.size())
    throw std::runtime_error("Vector dimensions are not the same in  GaussianRBF::eval_second_derivative(const vector_double x, const vector_double c)");
  
  auto hessian_at_x = std::make_shared< vector_double>(x.size() * x.size());

  double gaussian_at_x = eval(x, c);

  // above the diagonal and diagonal
  for(auto j = 0; j < x.size(); ++j) {
    for(auto k = j; k < x.size(); ++k) {
      hessian_at_x.get()->at(j * x.size() + k) = gaussian_at_x * (-2 * (x[j] - c[j]) / sigma_sqrd) * (-2 * (x[k] - c[k]) / sigma_sqrd);

      // adding term to the diagonal
      if(j == k)
        hessian_at_x.get()->at(j * x.size() + k) += gaussian_at_x * (-2 / sigma_sqrd);
      // copying the value to the transposed position if it is not part of the diagonal
      else
        hessian_at_x.get()->at(k * x.size() + j) = hessian_at_x.get()->at(j * x.size() + k);
    }
  }

  return hessian_at_x;
}
  
} // namespace tudat_learn