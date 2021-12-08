/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_RBF_TPP
#define TUDAT_LEARN_RBF_TPP

#ifndef TUDAT_LEARN_RBF_HPP
#ERROR __FILE__ should only be included from rbf.hpp
#endif

namespace tudat_learn
{

// CubicRBF //

template <typename T>
T CubicRBF<T>::eval(const T radius) const {
  return radius * radius * radius;
}

template <typename T>
typename CubicRBF<T>::MatrixXt CubicRBF<T>::eval_matrix(const MatrixXt &distance_matrix) const {
  return distance_matrix.array().cube().matrix();
}

template <typename T>
T CubicRBF<T>::eval(const vector_t &x, const vector_t &c) const {
  if(x.size() != c.size())
    throw std::runtime_error("Vector dimensions are not the same in   CubicRBF<T>::eval(const vector_t &x, const vector_t &c) const");

  T result = 0;

  for(auto j = 0; j < x.size(); ++j)
    result += (x[j] - c[j]) * (x[j] - c[j]);

  result = std::sqrt(result);
  result = result * result * result; // faster than pow(result, 3/2)

  return result;
}

template <typename T>
T CubicRBF<T>::eval(const VectorXt &x, const VectorXt &c) const {
  if(x.rows() != c.rows())
    throw std::runtime_error("Vector dimensions are not the same in CubicRBF<T>::eval(const VectorXt &x, const VectorXt &c) const");

  auto result = (x - c).norm();
  result = result * result * result;

  return result;
}

template <typename T>
std::shared_ptr< typename RBF<T>::vector_t > CubicRBF<T>::eval_jacobian(const vector_t &x, const vector_t &c) const {
  if( x.size() != c.size())
    throw std::runtime_error("Vector dimensions are not the same in CubicRBF<T>::eval_jacobian(const vector_t &x, const vector_t &c) const ");
  
  auto jacobian_at_x = std::make_shared<vector_t>();

  jacobian_at_x->reserve(x.size());
  
  T radius = 0;

  for(auto j = 0; j < x.size(); ++j)
    radius += (x[j] - c[j]) * (x[j] - c[j]);

  radius = std::sqrt(radius);

  for(auto j = 0; j < x.size(); ++j)
    jacobian_at_x.get()->push_back(3 * (x[j] - c[j]) * radius);

  return jacobian_at_x;
}

template <typename T>
std::shared_ptr< typename RBF<T>::VectorXt > CubicRBF<T>::eval_jacobian(const VectorXt &x, const VectorXt &c) const {
  if(x.rows() != c.rows())
    throw std::runtime_error("Vector dimensions are not the same in CubicRBF<T>::eval_jacobian(const VectorXt &x, const VectorXt &c) const)");
  
  auto jacobian_at_x = std::make_shared<VectorXt>(x - c);

  T radius = jacobian_at_x->norm();

  *jacobian_at_x *= 3 * radius;

  return jacobian_at_x;
}

template <typename T>
std::shared_ptr< typename RBF<T>::vector_t > CubicRBF<T>::eval_hessian(const vector_t &x, const vector_t &c) const {
  if(x.size() != c.size())
    throw std::runtime_error("Vector dimensions are not the same in CubicRBF<T>::eval_hessian(const vector_t &x, const vector_t &c) const");
  
  auto hessian_at_x = std::make_shared<vector_t>(x.size() * x.size());

  T radius = 0;

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

template <typename T>
std::shared_ptr< typename RBF<T>::MatrixXt > CubicRBF<T>::eval_hessian(const VectorXt &x, const VectorXt &c) const {
  if(x.rows() != c.rows())
    throw std::runtime_error("Vector dimensions are not the same in CubicRBF<T>::eval_hessian(const VectorXt &x, const VectorXt &c) const)");
  
  auto hessian_at_x = std::make_shared< MatrixXt >();

  hessian_at_x->resize(x.rows(), x.rows());

  auto vector_distance = x - c;

  *hessian_at_x = 3 * vector_distance * vector_distance.transpose() / vector_distance.norm();

  *hessian_at_x += 3 * vector_distance.norm() * MatrixXt::Identity(x.rows(), x.rows());

  return hessian_at_x;
}

// GaussianRBF //

template <typename T>
T GaussianRBF<T>::eval(const T radius) const {
  return std::exp(- (radius * radius) / (sigma_sqrd));
}

template <typename T>
typename GaussianRBF<T>::MatrixXt GaussianRBF<T>::eval_matrix(const MatrixXt &distance_matrix) const {
  return (distance_matrix.array().square() / (-sigma_sqrd)).exp().matrix();
}

template <typename T>
T GaussianRBF<T>::eval(const vector_t &x, const vector_t &c) const {
  if( x.size() != c.size())
    throw std::runtime_error("Vector dimensions are not the same in  GaussianRBF<T>::eval(const vector_t &x, const vector_t &c) const");

  T result = 0;

  for(auto j = 0; j < x.size(); ++j)
    result += (x[j] - c[j]) * (x[j] - c[j]);

  result = std::exp( - result / (sigma_sqrd));

  return result;
}

template <typename T>
T GaussianRBF<T>::eval(const VectorXt &x, const VectorXt &c) const {
  if(x.rows() != c.rows())
    throw std::runtime_error("Vector dimensions are not the same in GaussianRBF<T>::eval(const VectorXt &x, const VectorXt &c) const");

  auto result = (x - c).squaredNorm();
  result = std::exp(- result / sigma_sqrd);

  return result;
}

template <typename T>
std::shared_ptr< typename RBF<T>::vector_t > GaussianRBF<T>::eval_jacobian(const vector_t &x, const vector_t &c) const {
  if(x.size() != c.size())
    throw std::runtime_error("Vector dimensions are not the same in GaussianRBF<T>::eval_jacobian(const vector_t &x, const vector_t &c) const ");
  
  auto jacobian_at_x = std::make_shared<vector_t>();

  jacobian_at_x->reserve(x.size());
  
  T gaussian_at_x = eval(x, c);

  for(auto j = 0; j < x.size(); ++j)
    jacobian_at_x.get()->push_back(gaussian_at_x * (-2 * (x[j] - c[j]) / (sigma_sqrd)));

  return jacobian_at_x;
}

template <typename T>
std::shared_ptr< typename RBF<T>::VectorXt > GaussianRBF<T>::eval_jacobian(const VectorXt &x, const VectorXt &c) const {
  if(x.rows() != c.rows())
    throw std::runtime_error("Vector dimensions are not the same in GaussianRBF<T>::eval_jacobian(const VectorXt &x, const VectorXt &c) const)");
  
  auto jacobian_at_x = std::make_shared<VectorXt>(x - c);

  T gaussian_at_x = eval(x, c);

  *jacobian_at_x *= gaussian_at_x * -2 / sigma_sqrd;

  return jacobian_at_x;
}

template <typename T>
std::shared_ptr< typename RBF<T>::vector_t > GaussianRBF<T>::eval_hessian(const vector_t &x, const vector_t &c) const {
  if(x.size() != c.size())
    throw std::runtime_error("Vector dimensions are not the same in GaussianRBF<T>::eval_hessian(const vector_t &x, const vector_t &c) const");
  
  auto hessian_at_x = std::make_shared< vector_t>(x.size() * x.size());

  T gaussian_at_x = eval(x, c);

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

template <typename T>
std::shared_ptr< typename RBF<T>::MatrixXt > GaussianRBF<T>::eval_hessian(const VectorXt &x, const VectorXt &c) const {
  if(x.rows() != c.rows())
    throw std::runtime_error("Vector dimensions are not the same in GaussianRBF<T>::eval_hessian(const VectorXt &x, const VectorXt &c) const)");
  
  auto hessian_at_x = std::make_shared< MatrixXt >();

  hessian_at_x->resize(x.rows(), x.rows());

  auto gaussian_at_x = eval(x, c);

  auto vector_distance = -2 * (x - c) / sigma_sqrd;

  *hessian_at_x = gaussian_at_x * vector_distance * vector_distance.transpose();

  *hessian_at_x += gaussian_at_x * (-2 / sigma_sqrd) * MatrixXt::Identity(x.rows(), x.rows());

  return hessian_at_x;
}

  
} // namespace tudat_learn

#endif // TUDAT_LEARN_RBF_TPP