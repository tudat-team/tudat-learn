/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#include <iomanip>
#include <iostream>
#include <vector>

#include <Eigen/Core>

#include "tudat-learn/estimators/regressors/rbf.hpp"
#include "tudat-learn/estimators/regressors/rbfn.hpp"

int main( ) {
  
  // Values generated using /tudat-learn/cpp/tests/python_scripts/estimators/regressors/rbfn_test.py
  std::cout << std::setprecision(6) << std::fixed;

  std::vector< Eigen::VectorXf > data({
    (Eigen::VectorXf(7) << 0.548814, 0.715189, 0.602763, 0.544883, 0.423655, 0.645894, 0.437587).finished(),
    (Eigen::VectorXf(7) << 0.891773, 0.963663, 0.383442, 0.791725, 0.528895, 0.568045, 0.925597).finished(),
    (Eigen::VectorXf(7) << 0.071036, 0.087129, 0.020218, 0.832620, 0.778157, 0.870012, 0.978618).finished(),
    (Eigen::VectorXf(7) << 0.799159, 0.461479, 0.780529, 0.118274, 0.639921, 0.143353, 0.944669).finished(),
    (Eigen::VectorXf(7) << 0.521848, 0.414662, 0.264556, 0.774234, 0.456150, 0.568434, 0.018790).finished(),
    (Eigen::VectorXf(7) << 0.617635, 0.612096, 0.616934, 0.943748, 0.681820, 0.359508, 0.437032).finished(),
    (Eigen::VectorXf(7) << 0.697631, 0.060225, 0.666767, 0.670638, 0.210383, 0.128926, 0.315428).finished(),
    (Eigen::VectorXf(7) << 0.363711, 0.570197, 0.438602, 0.988374, 0.102045, 0.208877, 0.161310).finished(),
    (Eigen::VectorXf(7) << 0.653108, 0.253292, 0.466311, 0.244426, 0.158970, 0.110375, 0.656330).finished(),
    (Eigen::VectorXf(7) << 0.138183, 0.196582, 0.368725, 0.820993, 0.097101, 0.837945, 0.096098).finished()
  });
  
  std::vector< Eigen::VectorXf > labels({
    (Eigen::VectorXf(2) << 0.976459, 0.468651).finished(),
    (Eigen::VectorXf(2) << 0.976761, 0.604846).finished(),
    (Eigen::VectorXf(2) << 0.739264, 0.039188).finished(),
    (Eigen::VectorXf(2) << 0.282807, 0.120197).finished(),
    (Eigen::VectorXf(2) << 0.296140, 0.118728).finished(),
    (Eigen::VectorXf(2) << 0.317983, 0.414263).finished(),
    (Eigen::VectorXf(2) << 0.064147, 0.692472).finished(),
    (Eigen::VectorXf(2) << 0.566601, 0.265389).finished(),
    (Eigen::VectorXf(2) << 0.523248, 0.093941).finished(),
    (Eigen::VectorXf(2) << 0.575946, 0.929296).finished()
  });

  float sigma = 0.318569;

  auto dataset_ptr = std::make_shared< tudat_learn::Dataset<Eigen::VectorXf, Eigen::VectorXf> >(tudat_learn::Dataset(data, labels));
  auto cubic_rbf_ptr = std::make_shared< tudat_learn::CubicRBF<float> >(tudat_learn::CubicRBF<float>());
  auto gaussian_rbf_ptr = std::make_shared< tudat_learn::GaussianRBF<float> >(tudat_learn::GaussianRBF<float>(sigma));

  tudat_learn::RBFN<Eigen::VectorXf, Eigen::VectorXf> cubic_rbfn(dataset_ptr, cubic_rbf_ptr);
  cubic_rbfn.fit();
  std::cout << "Coefficients of the Cubic RBFN:\n" << cubic_rbfn.get_coefficients() << std::endl;

  Eigen::MatrixXf cubic_coefficients_expected(10,2);
  cubic_coefficients_expected <<  -1.337114, -2.755163,
                                  -0.212870, 2.093877,
                                  0.527279, -0.079324,
                                  -0.428405, -1.163986,
                                  1.476176, 4.215983,
                                  0.801391, -1.680252,
                                  0.322220, 0.782600,
                                  -0.267450, -1.772963,
                                  0.935899, 1.639483,
                                  -1.084187, -1.09602;

  if( !cubic_coefficients_expected.isApprox(cubic_rbfn.get_coefficients()) )
    return 1;

  tudat_learn::RBFN<Eigen::VectorXf, Eigen::VectorXf> gaussian_rbfn(dataset_ptr, gaussian_rbf_ptr);
  gaussian_rbfn.fit();
  std::cout << "Coefficients of the Gaussian RBFN:\n" << gaussian_rbfn.get_coefficients() << std::endl;

  Eigen::MatrixXf gaussian_coefficients_expected(10,2);
  gaussian_coefficients_expected <<  0.955820, 0.447495,
                                     0.971372, 0.601469,
                                     0.739263, 0.039187,
                                     0.278673, 0.119504,
                                     0.260680, 0.087732,
                                     0.269842, 0.390772,
                                     0.048714, 0.688877,
                                     0.557143, 0.256347,
                                     0.519257, 0.076439,
                                     0.570117, 0.927115;

  if( !gaussian_coefficients_expected.isApprox(gaussian_rbfn.get_coefficients()) )
    return 1;
                                  
  return 0;
}