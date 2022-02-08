/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#include <iostream>
#include <iomanip>

#include <Eigen/Core>

#include <tudat-learn/dataset.hpp>
#include <tudat-learn/estimators/regressors/rbf.hpp>
#include <tudat-learn/estimators/regressors/grnn.hpp>

int main() {
  
  // Values generated using /tudat-learn/cpp/tests/python_scripts/estimators/regressors/rbfn_test.py
  std::cout << std::setprecision(6) << std::fixed;

  std::vector< Eigen::VectorXf > center_points({
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

  auto dataset_ptr = std::make_shared< tudat_learn::Dataset<Eigen::VectorXf, Eigen::VectorXf> >(tudat_learn::Dataset(center_points, labels));
  auto gaussian_rbf_ptr = std::make_shared< tudat_learn::GaussianRBF<float> >(tudat_learn::GaussianRBF<float>(sigma));

  tudat_learn::GRNN<Eigen::VectorXf, Eigen::VectorXf> grnn(dataset_ptr, gaussian_rbf_ptr);
  grnn.fit();

  std::vector< Eigen::VectorXf > inputs({
    (Eigen::VectorXf(7) << 0.667410, 0.131798, 0.716327, 0.289406, 0.183191, 0.586513, 0.020108).finished(),
    (Eigen::VectorXf(7) << 0.828940, 0.004695, 0.677817, 0.270008, 0.735194, 0.962189, 0.248753).finished(),
    (Eigen::VectorXf(7) << 0.576157, 0.592042, 0.572252, 0.223082, 0.952749, 0.447125, 0.846409).finished()
  });

  Eigen::MatrixXf outputs(3,2);
  outputs = grnn.eval(inputs);
  std::cout << "GRNN output:\n" << outputs << std::endl;

  Eigen::MatrixXf expected_output(3, 2);
  expected_output << 0.212657, 0.569603,
                     0.712506, 0.374801,
                     0.327922, 0.146603;

  if( !expected_output.isApprox(outputs) )
    return 1;

  for(int i = 0; i < inputs.size(); ++i)
    if( !grnn.eval(inputs[i]).isApprox(expected_output.row(i).transpose()) )
      return 1;

  // Testing with Compile-time vectors
  using Vector7f = Eigen::Matrix<float, 7, 1>;
  std::vector< Vector7f > center_points_fixed({
    Vector7f(0.548814, 0.715189, 0.602763, 0.544883, 0.423655, 0.645894, 0.437587),
    Vector7f(0.891773, 0.963663, 0.383442, 0.791725, 0.528895, 0.568045, 0.925597),
    Vector7f(0.071036, 0.087129, 0.020218, 0.832620, 0.778157, 0.870012, 0.978618),
    Vector7f(0.799159, 0.461479, 0.780529, 0.118274, 0.639921, 0.143353, 0.944669),
    Vector7f(0.521848, 0.414662, 0.264556, 0.774234, 0.456150, 0.568434, 0.018790),
    Vector7f(0.617635, 0.612096, 0.616934, 0.943748, 0.681820, 0.359508, 0.437032),
    Vector7f(0.697631, 0.060225, 0.666767, 0.670638, 0.210383, 0.128926, 0.315428),
    Vector7f(0.363711, 0.570197, 0.438602, 0.988374, 0.102045, 0.208877, 0.161310),
    Vector7f(0.653108, 0.253292, 0.466311, 0.244426, 0.158970, 0.110375, 0.656330),
    Vector7f(0.138183, 0.196582, 0.368725, 0.820993, 0.097101, 0.837945, 0.096098)
  });
  
  std::vector< Eigen::Vector2f > labels_fixed({
    Eigen::Vector2f(0.976459, 0.468651),
    Eigen::Vector2f(0.976761, 0.604846),
    Eigen::Vector2f(0.739264, 0.039188),
    Eigen::Vector2f(0.282807, 0.120197),
    Eigen::Vector2f(0.296140, 0.118728),
    Eigen::Vector2f(0.317983, 0.414263),
    Eigen::Vector2f(0.064147, 0.692472),
    Eigen::Vector2f(0.566601, 0.265389),
    Eigen::Vector2f(0.523248, 0.093941),
    Eigen::Vector2f(0.575946, 0.929296)
  });

  auto dataset_ptr_fixed = std::make_shared< tudat_learn::Dataset<Vector7f, Eigen::Vector2f> >(tudat_learn::Dataset(center_points_fixed, labels_fixed));
  tudat_learn::GRNN<Vector7f, Eigen::Vector2f> grnn_fixed(dataset_ptr_fixed, gaussian_rbf_ptr);
  // tudat_learn::GRNN<Eigen::VectorXf, Eigen::VectorXf> grnn_fixed(dataset_ptr, gaussian_rbf_ptr);
  grnn_fixed.fit();

  std::vector< Vector7f > inputs_fixed({
    Vector7f(0.667410, 0.131798, 0.716327, 0.289406, 0.183191, 0.586513, 0.020108),
    Vector7f(0.828940, 0.004695, 0.677817, 0.270008, 0.735194, 0.962189, 0.248753),
    Vector7f(0.576157, 0.592042, 0.572252, 0.223082, 0.952749, 0.447125, 0.846409)
  });

  Eigen::MatrixXf outputs_fixed(3,2);
  outputs_fixed = grnn_fixed.eval(inputs_fixed);
  std::cout << "GRNN output fixed size:\n" << outputs_fixed << std::endl;

  if( !expected_output.isApprox(outputs_fixed) )
    return 1;

  for(int i = 0; i < inputs.size(); ++i)
    if( !grnn_fixed.eval(inputs[i]).isApprox(outputs_fixed.row(i).transpose()) )
      return 1;

  // Testing fit(const std::vector<int> &fit_indices)
  std::vector< Eigen::VectorXf > data_extra({
    (Eigen::VectorXf(7) << 1, 1, 2, 0, 0, 0, 0).finished(),
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
    (Eigen::VectorXf(7) << 0, 0, 0, 0, 0, 0, 0).finished()
  });

  std::vector< Eigen::VectorXf > labels_extra({
    (Eigen::VectorXf(2) << 1, 0).finished(),
    (Eigen::VectorXf(2) << 0.976459, 0.468651).finished(),
    (Eigen::VectorXf(2) << 0.976761, 0.604846).finished(),
    (Eigen::VectorXf(2) << 0.739264, 0.039188).finished(),
    (Eigen::VectorXf(2) << 0.282807, 0.120197).finished(),
    (Eigen::VectorXf(2) << 0.296140, 0.118728).finished(),
    (Eigen::VectorXf(2) << 0.317983, 0.414263).finished(),
    (Eigen::VectorXf(2) << 0.064147, 0.692472).finished(),
    (Eigen::VectorXf(2) << 0.566601, 0.265389).finished(),
    (Eigen::VectorXf(2) << 0.523248, 0.093941).finished(),
    (Eigen::VectorXf(2) << 0.575946, 0.929296).finished(),
    (Eigen::VectorXf(2) << 0, 0).finished()
  });

  auto dataset_extra_ptr = std::make_shared< tudat_learn::Dataset<Eigen::VectorXf, Eigen::VectorXf> >(tudat_learn::Dataset(data_extra, labels_extra));

  tudat_learn::GRNN<Eigen::VectorXf, Eigen::VectorXf> grnn_extra(dataset_extra_ptr, gaussian_rbf_ptr);
  grnn_extra.fit(std::vector<int>({1,2,3,4,5,6,7,8,9,10}));

  Eigen::MatrixXf outputs_extra(3,2);
  outputs_extra = grnn_extra.eval(inputs);
  std::cout << "GRNN output after fitting to certain indicies:\n" << outputs_extra << std::endl;

  if( !expected_output.isApprox(outputs_extra) )
    return 1;

  for(int i = 0; i < inputs.size(); ++i)
    if( !grnn_extra.eval(inputs[i]).isApprox(outputs_extra.row(i).transpose()) )
      return 1;

  return 0;
}