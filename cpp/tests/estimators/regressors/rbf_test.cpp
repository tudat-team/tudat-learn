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
#include <cmath>

#include <iostream>

#include <Eigen/Core>

#include "tudat-learn/estimators/regressors/rbf.hpp"
 
int main() {
  std::cout.precision(9);

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
  std::vector< double > gaussian_jacobian({-0.98767754, -0.41611931, -0.02637851});
  std::vector< double > gaussian_hessian({
    0.19575717,  0.79317606,  0.05028078,
    0.79317606, -1.35270747,  0.02118384,
    0.05028078,  0.02118384, -1.68553831
  });

  // Printing variables
  std::cout << "Variables:" << std::endl;
  std::cout << "x:" << std::endl;
  for(const auto &it : x)
    std::cout << it << ", ";
  std::cout << std::endl;
  std::cout << "c:" << std::endl;
  for(const auto &it : c)
    std::cout << it << ", ";
  std::cout << std::endl;
  std::cout << "sigma:\n" << sigma << std::endl;

  // Testing the CubicRBF
  std::cout << "\nTesting CubicRBF:" << std::endl;
  
  tudat_learn::CubicRBF<double> cubic_rbf;

  std::cout << "CubicRBF evaluated at x, c:" << std::endl;
  std::cout << cubic_rbf.eval(x, c) << std::endl;

  if(std::abs(cubic_rbf.eval(x, c) - cubic) > 1e-7 )
    return 1;


  auto cubic_jacobian_ptr = cubic_rbf.eval_jacobian(x, c);

  std::cout << "CubicRBF Jacobian evaluated at x, c:" << std::endl;
  for(auto i = 0; i < cubic_jacobian_ptr.get()->size(); ++i)
    std::cout << cubic_jacobian_ptr.get()->at(i) << ", ";
  std::cout << std::endl;

  if(cubic_jacobian_ptr.get()->size() != cubic_jacobian.size())
    return 1;

  for(auto i = 0; i < cubic_jacobian_ptr.get()->size(); ++i)
    if(std::abs(cubic_jacobian_ptr.get()->at(i) - cubic_jacobian.at(i)) > 1e-7 )
      return 1;  


  auto cubic_hessian_ptr = cubic_rbf.eval_hessian(x, c);

  std::cout << "CubicRBF Hessian evaluated at x, c:" << std::endl;
  for(auto i = 0; i < cubic_hessian_ptr.get()->size(); ++i)
    std::cout << cubic_hessian_ptr.get()->at(i) << ", ";
  std::cout << std::endl;

  if(cubic_hessian_ptr.get()->size() != cubic_hessian.size())
    return 1;

  for(auto i = 0; i < cubic_hessian_ptr.get()->size(); ++i)
    if(std::abs(cubic_hessian_ptr.get()->at(i) - cubic_hessian.at(i)) > 1e-7 )
      return 1;

  

  std::cout << "\nTesting GaussianRBF(sigma):" << std::endl;

  tudat_learn::GaussianRBF<double> gaussian_rbf(sigma);

  std::cout << "GaussianRBF(sigma) evaluated at x, c:" << std::endl;
  std::cout << gaussian_rbf.eval(x, c) << std::endl;

  if(std::abs(gaussian_rbf.eval(x, c) - gaussian) > 1e-7 )
    return 1;

  auto gaussian_jacobian_ptr = gaussian_rbf.eval_jacobian(x, c);

  std::cout << "GaussianRBF(sigma) Jacobian evaluated at x, c:" << std::endl;
  for(auto i = 0; i < gaussian_jacobian_ptr.get()->size(); ++i)
    std::cout << gaussian_jacobian_ptr.get()->at(i) << ", ";
  std::cout << std::endl;

  if(gaussian_jacobian_ptr.get()->size() != gaussian_jacobian.size())
    return 1;
    

  for(auto i = 0; i < gaussian_jacobian_ptr.get()->size(); ++i)
    if(std::abs(gaussian_jacobian_ptr.get()->at(i) - gaussian_jacobian.at(i)) > 1e-7 )
      return 1;  

  auto gaussian_hessian_ptr = gaussian_rbf.eval_hessian(x, c);

  std::cout << "GaussianRBF(sigma) Hessian evaluated at x, c:" << std::endl;
  for(auto i = 0; i < gaussian_hessian_ptr.get()->size(); ++i)
    std::cout << gaussian_hessian_ptr.get()->at(i) << ", ";
  std::cout << std::endl;

  if(gaussian_hessian_ptr.get()->size() != gaussian_hessian.size())
    return 1;

  for(auto i = 0; i < gaussian_hessian_ptr.get()->size(); ++i)
    if(std::abs(gaussian_hessian_ptr.get()->at(i) - gaussian_hessian.at(i)) > 1e-7 )
      return 1;


  return 0;
}


