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
std::shared_ptr< typename RBF<T>::vector_t > CubicRBF<T>::eval_gradient(const vector_t &x, const vector_t &c) const {
  if( x.size() != c.size())
    throw std::runtime_error("Vector dimensions are not the same in CubicRBF<T>::eval_gradient(const vector_t &x, const vector_t &c) const ");
  
  auto gradient_at_x = std::make_shared<vector_t>();

  gradient_at_x->reserve(x.size());
  
  T radius = 0;

  for(auto j = 0; j < x.size(); ++j)
    radius += (x[j] - c[j]) * (x[j] - c[j]);

  radius = std::sqrt(radius);

  for(auto j = 0; j < x.size(); ++j)
    gradient_at_x.get()->push_back(3 * (x[j] - c[j]) * radius);

  return gradient_at_x;
}

template <typename T>
std::shared_ptr< typename RBF<T>::VectorXt > CubicRBF<T>::eval_gradient(const VectorXt &x, const VectorXt &c) const {
  if(x.rows() != c.rows())
    throw std::runtime_error("Vector dimensions are not the same in CubicRBF<T>::eval_gradient(const VectorXt &x, const VectorXt &c) const)");
  
  auto gradient_at_x = std::make_shared<VectorXt>(x - c);

  T radius = gradient_at_x->norm();

  *gradient_at_x *= 3 * radius;

  return gradient_at_x;
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

template <typename T>
typename RBF<T>::MatrixXt CubicRBF<T>::gradient_rbfn(const VectorXt &x, const MatrixXt &center_points) const {
  
  // This matrix contains the element-wise difference between the input vector (x) and every center point(c)
  // Its dimensions are [#(center points) x dimension_input] = N x di and is of the following form:
  // (x1 - c11) (x2 - c12) ... (xdi - c1di)
  // (x1 - c21) (x2 - c22) ... (xdi - c2di) 
  //    ...        ...     ...     ...
  // (x1 - cN1) (x2 - cN2) ... (xdi - cNdi) 
  MatrixXt difference_input_center_points(
    center_points.rows(), center_points.cols()
  );

  difference_input_center_points = center_points.rowwise() - x.transpose();

  // This matrix contains the Euclidean distances between the input and every center point as inputs.
  // Dimensions: [#(center points) x 1]
  VectorXt distance_matrix(
    center_points.rows()
  );

  distance_matrix = difference_input_center_points.rowwise().norm();

  // Matrix with partial derivatives that when multiplied by the coefficients of the RBFN yields its gradient
  // as explained in the description of this function in the header file.
  MatrixXt partial_derivatives(
    center_points.rows(), center_points.cols()
  );

  partial_derivatives = 3 * (difference_input_center_points.array().colwise() * distance_matrix.array()).matrix();

  return partial_derivatives;
}

template <typename T>
std::vector<typename RBF<T>::MatrixXt> CubicRBF<T>::hessian_rbfn(const VectorXt &x, const MatrixXt &center_points) const {
  // This matrix contains the element-wise difference between the input vector (x) and every center point(c)
  // Its dimensions are [#(center points) x dimension_input] = N x di and is of the following form:
  // (x1 - c11) (x2 - c12) ... (xdi - c1di)
  // (x1 - c21) (x2 - c22) ... (xdi - c2di) 
  //    ...        ...     ...     ...
  // (x1 - cN1) (x2 - cN2) ... (xdi - cNdi) 
  MatrixXt difference_input_center_points(
    center_points.rows(), center_points.cols()
  );

  difference_input_center_points = center_points.rowwise() - x.transpose();

  // This matrix contains the Euclidean distances between the input and every center point as inputs.
  // Dimensions: [#(center points) x 1]
  VectorXt distance_matrix(
    center_points.rows()
  );

  distance_matrix = difference_input_center_points.rowwise().norm();

  // Contains multiple matrices of second order partial derivatives of the RBFs with respect to different
  // input variables and for the various center-points. When each of these matrices is multiplied by the 
  // coefficients from one of the RBFN's output dimensions, it yields a row of the RBFN's hessian matrix for
  // that respective output dimension.
  // Let di = dimension_input; N = #(center points); d2f = second order derivative of RBF;
  // Matrix k of the vector has dimensions [#(center points) x dimension_input] and the following form:
  // d2f/(dxk dx1)|c1   d2f/(dxk dx2)|c1  ...   d2f/(dxk dxdi)|c1
  // d2f/(dxk dx1)|c2   d2f/(dxk dx2)|c2  ...   d2f/(dxk dxdi)|c2
  //      ...                 ...         ...         ...
  // d2f/(dxk dx1)|cN   d2f/(dxk dx2)|cN  ...   d2f/(dxk dxdi)|cN
  // When multiplied by the transposed column i-th column of the coefficients cf: cf.col(i).transpose() * Matrix k
  // the result is the k-th row of the Hessian matrix of the RBFN that concerns the i-th output dimension.
  std::vector<MatrixXt> rbf_second_order_derivatives;
  rbf_second_order_derivatives.reserve(center_points.cols());

  for(int k = 0; k < center_points.cols(); ++k){
    MatrixXt k_th_derivative_matrix(
      center_points.rows(), center_points.cols()
    );

    k_th_derivative_matrix = 3 * (
      difference_input_center_points.array().colwise() * 
      difference_input_center_points.col(k).array()    * 
      distance_matrix.array()
    ).matrix();

    k_th_derivative_matrix.col(k) = k_th_derivative_matrix.col(k) + 3 * distance_matrix;

    rbf_second_order_derivatives.push_back(k_th_derivative_matrix);
  }

  return rbf_second_order_derivatives;
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
std::shared_ptr< typename RBF<T>::vector_t > GaussianRBF<T>::eval_gradient(const vector_t &x, const vector_t &c) const {
  if(x.size() != c.size())
    throw std::runtime_error("Vector dimensions are not the same in GaussianRBF<T>::eval_gradient(const vector_t &x, const vector_t &c) const ");
  
  auto gradient_at_x = std::make_shared<vector_t>();

  gradient_at_x->reserve(x.size());
  
  T gaussian_at_x = eval(x, c);

  for(auto j = 0; j < x.size(); ++j)
    gradient_at_x.get()->push_back(gaussian_at_x * (-2 * (x[j] - c[j]) / (sigma_sqrd)));

  return gradient_at_x;
}

template <typename T>
std::shared_ptr< typename RBF<T>::VectorXt > GaussianRBF<T>::eval_gradient(const VectorXt &x, const VectorXt &c) const {
  if(x.rows() != c.rows())
    throw std::runtime_error("Vector dimensions are not the same in GaussianRBF<T>::eval_gradient(const VectorXt &x, const VectorXt &c) const)");
  
  auto gradient_at_x = std::make_shared<VectorXt>(x - c);

  T gaussian_at_x = eval(x, c);

  *gradient_at_x *= gaussian_at_x * -2 / sigma_sqrd;

  return gradient_at_x;
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

template <typename T>
typename RBF<T>::MatrixXt GaussianRBF<T>::gradient_rbfn(const VectorXt &x, const MatrixXt &center_points) const {
  
  // This matrix contains the element-wise difference between the input vector (x) and every center point(c)
  // Its dimensions are [#(center points) x dimension_input] = N x di and is of the following form:
  // (x1 - c11) (x2 - c12) ... (xdi - c1di)
  // (x1 - c21) (x2 - c22) ... (xdi - c2di) 
  //    ...        ...     ...     ...
  // (x1 - cN1) (x2 - cN2) ... (xdi - cNdi) 
  MatrixXt difference_input_center_points(
    center_points.rows(), center_points.cols()
  );

  difference_input_center_points = center_points.rowwise() - x.transpose();

  // This matrix contains the output of the RBF when given the Euclidean distances between the input and every
  // center point as inputs.
  // Dimensions [#(center points) x 1]
  VectorXt rbf_matrix(
    center_points.rows()
  );

  rbf_matrix = this->eval_matrix(difference_input_center_points.rowwise().norm());

  // Matrix with partial derivatives that when multiplied by the coefficients of the RBFN yields its gradient
  // as explained in the description of this function in the header file.
  MatrixXt partial_derivatives(
    center_points.rows(), center_points.cols()
  );

  partial_derivatives = (-2 / sigma_sqrd) * (difference_input_center_points.array().colwise() * rbf_matrix.array()).matrix();

  return partial_derivatives;
}

template <typename T>
std::vector<typename RBF<T>::MatrixXt>  GaussianRBF<T>::hessian_rbfn(const VectorXt &x, const MatrixXt &center_points) const {
  // This matrix contains the element-wise difference between the input vector (x) and every center point(c)
  // Its dimensions are [#(center points) x dimension_input] = N x di and is of the following form:
  // (x1 - c11) (x2 - c12) ... (xdi - c1di)
  // (x1 - c21) (x2 - c22) ... (xdi - c2di) 
  //    ...        ...     ...     ...
  // (x1 - cN1) (x2 - cN2) ... (xdi - cNdi) 
  MatrixXt difference_input_center_points(
    center_points.rows(), center_points.cols()
  );

  difference_input_center_points = center_points.rowwise() - x.transpose();

  // This matrix contains the Euclidean distances between the input and every center point as inputs.
  // Dimensions: [#(center points) x 1]
  VectorXt rbf_output_vector(
    center_points.rows()
  );

  rbf_output_vector = this->eval_matrix(difference_input_center_points.rowwise().norm());

  // Contains multiple matrices of second order partial derivatives of the RBFs with respect to different
  // input variables and for the various center-points. When each of these matrices is multiplied by the 
  // coefficients from one of the RBFN's output dimensions, it yields a row of the RBFN's hessian matrix for
  // that respective output dimension.
  // Let di = dimension_input; N = #(center points); d2f = second order derivative of RBF;
  // Matrix k of the vector has dimensions [#(center points) x dimension_input] and the following form:
  // d2f/(dxk dx1)|c1   d2f/(dxk dx2)|c1  ...   d2f/(dxk dxdi)|c1
  // d2f/(dxk dx1)|c2   d2f/(dxk dx2)|c2  ...   d2f/(dxk dxdi)|c2
  //      ...                 ...         ...         ...
  // d2f/(dxk dx1)|cN   d2f/(dxk dx2)|cN  ...   d2f/(dxk dxdi)|cN
  // When multiplied by the transposed column i-th column of the coefficients cf: cf.col(i).transpose() * Matrix k
  // the result is the k-th row of the Hessian matrix of the RBFN that concerns the i-th output dimension.
  std::vector<MatrixXt> rbf_second_order_derivatives;
  rbf_second_order_derivatives.reserve(center_points.cols());

  for(int k = 0; k < center_points.cols(); ++k){
    MatrixXt k_th_derivative_matrix(
      center_points.rows(), center_points.cols()
    );

    k_th_derivative_matrix = (-2 / sigma_sqrd) * (-2 / sigma_sqrd) * (
      difference_input_center_points.array().colwise() * 
      difference_input_center_points.col(k).array()    * 
      rbf_output_vector.array()
    ).matrix();

    k_th_derivative_matrix.col(k) = k_th_derivative_matrix.col(k) + (-2 / sigma_sqrd) * rbf_output_vector;

    rbf_second_order_derivatives.push_back(k_th_derivative_matrix);
  }

  return rbf_second_order_derivatives;
}
  
} // namespace tudat_learn

#endif // TUDAT_LEARN_RBF_TPP