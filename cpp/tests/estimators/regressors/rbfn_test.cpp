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

#include <Eigen/Core>

#include "tudat-learn/estimators/regressors/rbfn.hpp"

int main( ) {
  
  Eigen::VectorXf vec1(5); 
  vec1 << 1.0, 2.0, 3.0, 2.0, 1.0;

  Eigen::VectorXf vec2(5); 
  vec2 << 2.0, 4.0, 6.0, 8.0, 10.0;

  std::vector< Eigen::VectorXf > data({
    (Eigen::VectorXf(5) << 1.0, 2.0, 3.0, 2.0, 1.0).finished(),
    (Eigen::VectorXf(5) << 2.0, 4.0, 6.0, 8.0, 10.0).finished(),
  });

  std::vector<float> labels({
    1, 2
  });

  auto dataset_ptr = std::make_shared< tudat_learn::Dataset<Eigen::VectorXf, float> >(tudat_learn::Dataset(data, labels));
  auto rbf_ptr = std::make_shared< tudat_learn::CubicRBF<float> >(tudat_learn::CubicRBF<float>( ));

  tudat_learn::RBFN<Eigen::VectorXf, float> rbfn(dataset_ptr, rbf_ptr);

  return 0;
}