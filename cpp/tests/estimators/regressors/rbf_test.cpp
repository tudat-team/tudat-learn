/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#include <vector>
#include <memory>

#include <iostream>

#include "estimators/regressors/rbf.h"
 
int main() {

  // values obtained using cpp/tests/estimators/regressors/rbf_test.cpp

  std::vector< double > x({0.84442185, 0.7579544,  0.42057158});
  std::vector< double > c({0.25891675, 0.51127472, 0.40493414});

  double sigma = 0.7837985890347726;

  double cubic = 0.25670216;
  std::vector< double > cubic_jacobian({1.11633646, 0.47032472, 0.02981468});
  std::vector< double > cubic_hessian({
    3.52484826, 0.68177668, 0.04321898,
    0.68177668, 2.19386119, 0.01820863,
    0.04321898, 0.01820863, 1.90777552
  });

  double gaussian = 0.51815949;
  std::vector< double > gaussian_jacobian({-0.98767754 -0.41611931 -0.02637851});
  std::vector< double > gaussian_hessian({
    0.19575717,  0.79317606,  0.05028078,
    0.79317606, -1.35270747,  0.02118384,
    0.05028078,  0.02118384, -1.68553831
  });

  tudat_learn::CubicRBF cubic_rbf;
  tudat_learn::GaussianRBF gaussian_rbf(sigma);


  std::cout.precision(9);
  std::cout << cubic_rbf.eval(x, c) << std::endl;

  auto cubic_jacobian_ptr = cubic_rbf.eval_jacobian(x, c);

  for(auto i = 0; i < cubic_jacobian_ptr.get()->size(); ++i)
    std::cout << cubic_jacobian_ptr.get()->at(i) << ", ";
  std::cout << std::endl;

  auto cubic_hessian_ptr = cubic_rbf.eval_hessian(x, c);
  for(auto i = 0; i < cubic_hessian_ptr.get()->size(); ++i)
    std::cout << cubic_hessian_ptr.get()->at(i) << ", ";
  std::cout << std::endl;

  auto gaussian_jacobian_ptr = gaussian_rbf.eval_jacobian(x, c);

  for(auto i = 0; i < gaussian_jacobian_ptr.get()->size(); ++i)
    std::cout << gaussian_jacobian_ptr.get()->at(i) << ", ";
  std::cout << std::endl;

  auto gaussian_hessian_ptr = gaussian_rbf.eval_hessian(x, c);
  for(auto i = 0; i < gaussian_hessian_ptr.get()->size(); ++i)
    std::cout << gaussian_hessian_ptr.get()->at(i) << ", ";
  std::cout << std::endl;

  return 0;
}

// x is  [0.84442185 0.7579544  0.42057158]
// c is  [0.25891675 0.51127472 0.40493414]
// sigma is  0.7837985890347726
// Cubic evaluated at x and c is:  [[0.25670216]]
// Cubic Jacobian evaluated at x and c is:  [[1.11633646 0.47032472 0.02981468]]
// Cubic Hessian evaluated at x and c is:
//  [[3.52484826 0.68177668 0.04321898]
//  [0.68177668 2.19386119 0.01820863]
//  [0.04321898 0.01820863 1.90777552]]
// Gaussian evaluated at x, c and sigma is:  [[0.51815949]]
// Gaussian Jacobian evaluated at x, c and sigma is:  [[-0.98767754 -0.41611931 -0.02637851]]
// Gaussian Hessian evaluated at x, c and sigma is:
//  [[ 0.19575717  0.79317606  0.05028078]
//  [ 0.79317606 -1.35270747  0.02118384]
//  [ 0.05028078  0.02118384 -1.68553831]]


