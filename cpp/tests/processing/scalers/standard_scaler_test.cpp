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
#include <memory>
#include <vector>

#include <Eigen/Core>

#include "tudat-learn/processing/scalers/standard_scaler.hpp"

int main() {
  std::vector<Eigen::VectorXf> data_dynamic_vector({
    (Eigen::VectorXf(7) << 0.548814, 0.715189, 0.602763, 0.544883, 0.423655, 0.645894, 0.437587).finished(),
    (Eigen::VectorXf(7) << 0.891773, 0.963663, 0.383442, 0.791725, 0.528895, 0.568045, 0.925597).finished(),
    (Eigen::VectorXf(7) << 0.071036, 0.087129, 0.020218, 0.832620, 0.778157, 0.870012, 0.978618).finished(),
    (Eigen::VectorXf(7) << 0.799159, 0.461479, 0.780529, 0.118274, 0.639921, 0.143353, 0.944669).finished(),
    (Eigen::VectorXf(7) << 0.521848, 0.414662, 0.264556, 0.774234, 0.456150, 0.568434, 0.018790).finished(),
    (Eigen::VectorXf(7) << 0.617635, 0.612096, 0.616934, 0.943748, 0.681820, 0.359508, 0.437032).finished(),
    (Eigen::VectorXf(7) << 0.697631, 0.060225, 0.666767, 0.670638, 0.210383, 0.128926, 0.315428).finished(),
    (Eigen::VectorXf(7) << 0.363711, 0.570197, 0.438602, 0.988374, 0.102045, 0.208877, 0.161310).finished(),
    (Eigen::VectorXf(7) << 0.653108, 0.253292, 0.466311, 0.244426, 0.158970, 0.110375, 0.656330).finished(),
    (Eigen::VectorXf(7) << 0.138183, 0.196582, 0.368725, 0.820993, 0.097101, 0.837945, 0.096098).finished(),
  });

  std::vector<int> labels({
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  });

  auto dataset_ptr = std::make_shared< tudat_learn::Dataset<Eigen::VectorXf, int> >(tudat_learn::Dataset(data_dynamic_vector, labels));

  tudat_learn::StandardScaler<Eigen::VectorXf, int> scaler_eigen;
  scaler_eigen.fit(*dataset_ptr);

  auto expected_mean = (Eigen::VectorXf(7) << 0.530290, 0.433451, 0.460885, 0.672991, 0.407710, 0.444137, 0.497146).finished();
  std::cout << "Mean:\n" << scaler_eigen.get_mean().transpose() << std::endl;
  if( !expected_mean.isApprox(scaler_eigen.get_mean()) )
    return 1;

  auto expected_std = (Eigen::VectorXf(7) << 0.254264, 0.275691, 0.208995, 0.274282, 0.239644, 0.277795, 0.344040).finished();
  std::cout << "Standard Deviation:\n" << scaler_eigen.get_standard_deviation().transpose() << std::endl;
  if( !expected_std.isApprox(scaler_eigen.get_standard_deviation()) )
    return 1;

  auto expected_variance = (Eigen::VectorXf(7) << 0.064650, 0.076005, 0.043679, 0.075231, 0.057429, 0.077170, 0.118363).finished();
  std::cout << "Variance:\n" << scaler_eigen.get_variance().transpose() << std::endl;
  if( !expected_variance.isApprox(scaler_eigen.get_variance()) )
    return 1;

  // fixed vector
  // matrix
  // array
  // arithmetic

  // tudat_learn::StandardScaler<float, float> scaler_arithmetic;


  return 0;
}