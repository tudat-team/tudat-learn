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
#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include <Eigen/Core>

#include <tudat-learn/processing/scalers/standard_scaler.hpp>

int main() {
  // Values generated using /tudat-learn/cpp/tests/python_scripts/processing/scalers/standard_scaler_test.py
  std::cout << std::setprecision(6) << std::fixed;

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

  std::vector<Eigen::VectorXf> expected_scaled_data{
    (Eigen::VectorXf(7) <<  0.072854,  1.021933,  0.678859, -0.467069,  0.066537,  0.726282, -0.173116).finished(),
    (Eigen::VectorXf(7) <<  1.421684,  1.923211, -0.370548,  0.432889,  0.505688,  0.446042,  1.245353).finished(),
    (Eigen::VectorXf(7) << -1.806208, -1.256199, -2.108502,  0.581987,  1.545821,  1.533057,  1.399466).finished(),
    (Eigen::VectorXf(7) <<  1.057441,  0.101663,  1.529434, -2.022435,  0.968983, -1.082756,  1.300789).finished(),
    (Eigen::VectorXf(7) << -0.033201, -0.068154, -0.939394,  0.369118,  0.202134,  0.447442, -1.390408).finished(),
    (Eigen::VectorXf(7) <<  0.343522,  0.647989,  0.746665,  0.987147,  1.143821, -0.304646, -0.174729).finished(),
    (Eigen::VectorXf(7) <<  0.658139, -1.353786,  0.985106, -0.008581, -0.823415, -1.134690, -0.528189).finished(),
    (Eigen::VectorXf(7) << -0.655141,  0.496011, -0.106618,  1.149848, -1.275493, -0.846884, -0.976154).finished(),
    (Eigen::VectorXf(7) <<  0.483034, -0.653484,  0.025964, -1.562500, -1.037953, -1.201470,  0.462691).finished(),
    (Eigen::VectorXf(7) << -1.542124, -0.859185, -0.440966,  0.539596, -1.296123,  1.417623, -1.165702).finished(),
  };
  auto scaled_dataset(scaler_eigen.transform(*dataset_ptr));
  std::cout << "Scaled Dataset:" << std::endl;
  for(std::size_t i = 0; i < scaled_dataset.size(); ++i) {
    std::cout << scaled_dataset.data_at(i).transpose() << std::endl;
    if( !expected_scaled_data.at(i).isApprox(scaled_dataset.data_at(i)))
      return 1;
  }

  for(std::size_t i = 0; i < scaled_dataset.size(); ++i)
    if( !data_dynamic_vector.at(i).isApprox(scaler_eigen.inverse_transform(scaled_dataset.data_at(i))))
      return 1;

  // Same test with an array of fixed size.
  using Array7f = Eigen::Array<float, 7, 1>;
  std::vector<Array7f> data_static_array({
    Array7f(0.548814, 0.715189, 0.602763, 0.544883, 0.423655, 0.645894, 0.437587),
    Array7f(0.891773, 0.963663, 0.383442, 0.791725, 0.528895, 0.568045, 0.925597),
    Array7f(0.071036, 0.087129, 0.020218, 0.832620, 0.778157, 0.870012, 0.978618),
    Array7f(0.799159, 0.461479, 0.780529, 0.118274, 0.639921, 0.143353, 0.944669),
    Array7f(0.521848, 0.414662, 0.264556, 0.774234, 0.456150, 0.568434, 0.018790),
    Array7f(0.617635, 0.612096, 0.616934, 0.943748, 0.681820, 0.359508, 0.437032),
    Array7f(0.697631, 0.060225, 0.666767, 0.670638, 0.210383, 0.128926, 0.315428),
    Array7f(0.363711, 0.570197, 0.438602, 0.988374, 0.102045, 0.208877, 0.161310),
    Array7f(0.653108, 0.253292, 0.466311, 0.244426, 0.158970, 0.110375, 0.656330),
    Array7f(0.138183, 0.196582, 0.368725, 0.820993, 0.097101, 0.837945, 0.096098),
  });

  auto dataset_ptr_static_array = std::make_shared< tudat_learn::Dataset<Array7f, int> >(tudat_learn::Dataset(data_static_array, labels));

  tudat_learn::StandardScaler<Array7f, int> scaler_static_array;
  scaler_static_array.fit(*dataset_ptr_static_array);

  auto expected_mean_static_array = Array7f(0.530290, 0.433451, 0.460885, 0.672991, 0.407710, 0.444137, 0.497146);
  std::cout << "Mean Static Array:\n" << scaler_static_array.get_mean().transpose() << std::endl;
  if( !expected_mean_static_array.isApprox(scaler_static_array.get_mean()) )
    return 1;

  auto expected_std_static_array = Array7f(0.254264, 0.275691, 0.208995, 0.274282, 0.239644, 0.277795, 0.344040);
  std::cout << "Standard Deviation Static Array:\n" << scaler_static_array.get_standard_deviation().transpose() << std::endl;
  if( !expected_std_static_array.isApprox(scaler_static_array.get_standard_deviation()) )
    return 1;

  auto expected_variance_static_array = Array7f(0.064650, 0.076005, 0.043679, 0.075231, 0.057429, 0.077170, 0.118363);
  std::cout << "Variance Static Array:\n" << scaler_static_array.get_variance().transpose() << std::endl;
  if( !expected_variance_static_array.isApprox(scaler_static_array.get_variance()) )
    return 1;

  std::vector<Array7f> expected_scaled_data_static_array{
    Array7f( 0.072854,  1.021933,  0.678859, -0.467069,  0.066537,  0.726282, -0.173116),
    Array7f( 1.421684,  1.923211, -0.370548,  0.432889,  0.505688,  0.446042,  1.245353),
    Array7f(-1.806208, -1.256199, -2.108502,  0.581987,  1.545821,  1.533057,  1.399466),
    Array7f( 1.057441,  0.101663,  1.529434, -2.022435,  0.968983, -1.082756,  1.300789),
    Array7f(-0.033201, -0.068154, -0.939394,  0.369118,  0.202134,  0.447442, -1.390408),
    Array7f( 0.343522,  0.647989,  0.746665,  0.987147,  1.143821, -0.304646, -0.174729),
    Array7f( 0.658139, -1.353786,  0.985106, -0.008581, -0.823415, -1.134690, -0.528189),
    Array7f(-0.655141,  0.496011, -0.106618,  1.149848, -1.275493, -0.846884, -0.976154),
    Array7f( 0.483034, -0.653484,  0.025964, -1.562500, -1.037953, -1.201470,  0.462691),
    Array7f(-1.542124, -0.859185, -0.440966,  0.539596, -1.296123,  1.417623, -1.165702),
  };
  auto scaled_dataset_static_array(scaler_static_array.transform(*dataset_ptr_static_array));
  std::cout << "Scaled Dataset Static Array:" << std::endl;
  for(std::size_t i = 0; i < scaled_dataset_static_array.size(); ++i) {
    std::cout << scaled_dataset_static_array.data_at(i).transpose() << std::endl;
    if( !expected_scaled_data_static_array.at(i).isApprox(scaled_dataset_static_array.data_at(i)))
      return 1;
  }

  for(std::size_t i = 0; i < scaled_dataset_static_array.size(); ++i)
    if( !data_static_array.at(i).isApprox(scaler_static_array.inverse_transform(scaled_dataset_static_array.data_at(i))))
      return 1;

  // Test with scalar values.
  std::vector<float> data_scalar({
    0.976459, 0.468651, 0.976761, 0.604846, 0.739264, 0.039188, 0.282807, 0.120197, 0.296140, 0.118728
  });

  auto dataset_ptr_scalar = std::make_shared< tudat_learn::Dataset<float, int> >(tudat_learn::Dataset(data_scalar, labels));

  tudat_learn::StandardScaler<float, int> scaler_scalar;
  scaler_scalar.fit(*dataset_ptr_scalar);

  float expected_mean_scalar = 0.462304f;
  std::cout << "Mean Scalar Array:\n" << scaler_scalar.get_mean() << std::endl;
  if( std::abs(expected_mean_scalar - scaler_scalar.get_mean()) > 1e-6 )
    return 1;

  float expected_std_scalar = 0.331666f;
  std::cout << "Standard Deviation Scalar Array:\n" << scaler_scalar.get_standard_deviation() << std::endl;
  if( std::abs(expected_std_scalar - scaler_scalar.get_standard_deviation()) > 1e-6 )
    return 1;

  float expected_variance_scalar = 0.110003f;
  std::cout << "Variance Scalar Array:\n" << scaler_scalar.get_variance() << std::endl;
  if( std::abs(expected_variance_scalar - scaler_scalar.get_variance()) > 1e-6 )
    return 1;

  std::vector<float> expected_scaled_data_scalar{
    1.550219, 0.019137, 1.551130, 0.429776, 0.835057, -1.275729, -0.541198, -1.031480, -0.500998, -1.035910
  };
  auto scaled_dataset_scalar(scaler_scalar.transform(*dataset_ptr_scalar));
  std::cout << "Scaled Dataset Scalar:" << std::endl;
  for(std::size_t i = 0; i < scaled_dataset_scalar.size(); ++i) {
    std::cout << scaled_dataset_scalar.data_at(i) << std::endl;
    if( std::abs(expected_scaled_data_scalar.at(i) - scaled_dataset_scalar.data_at(i)) > 1e-5 )
      return 1;
  }

  for(std::size_t i = 0; i < scaled_dataset_scalar.size(); ++i)
    if( std::abs(data_scalar.at(i) - scaler_scalar.inverse_transform(scaled_dataset_scalar.data_at(i))) > 1e-6 )
      return 1;

  // Test with Static Matrices.
  std::vector<Eigen::Matrix2f> data_static_matrix({
    Eigen::Matrix2f({{0.317983, 0.414263},
                     {0.064147, 0.692472}}),
    Eigen::Matrix2f({{0.566601, 0.265389},
                     {0.523248, 0.093941}}),
    Eigen::Matrix2f({{0.575946, 0.929296},
                     {0.318569, 0.667410}}),
    Eigen::Matrix2f({{0.131798, 0.716327},
                     {0.289406, 0.183191}})
  });

  std::vector<int> labels_matrix_data({
    0,0,0,0
  });

  auto dataset_ptr_static_matrix = std::make_shared< tudat_learn::Dataset<Eigen::Matrix2f, int> >(tudat_learn::Dataset(data_static_matrix, labels_matrix_data));

  tudat_learn::StandardScaler<Eigen::Matrix2f, int> scaler_static_matrix;
  scaler_static_matrix.fit(*dataset_ptr_static_matrix);

  auto expected_mean_static_matrix = Eigen::Matrix2f({{0.398082, 0.581319},
                                                      {0.298842, 0.409253}});
  std::cout << "Mean Static Matrix:\n" << scaler_static_matrix.get_mean().transpose() << std::endl;
  if( !expected_mean_static_matrix.isApprox(scaler_static_matrix.get_mean()) )
    return 1;

  auto expected_std_static_matrix = Eigen::Matrix2f({{0.185309, 0.258377},
                                                     {0.162725, 0.272665}});
  std::cout << "Standard Deviation Static Matrix:\n" << scaler_static_matrix.get_standard_deviation().transpose() << std::endl;
  if( !expected_std_static_matrix.isApprox(scaler_static_matrix.get_standard_deviation()) )
    return 1;

  auto expected_variance_static_matrix = Eigen::Matrix2f({{0.034339, 0.066759},
                                                          {0.026479, 0.074346}});
  std::cout << "Variance Static Matrix:\n" << scaler_static_matrix.get_variance().transpose() << std::endl;
  if( !expected_variance_static_matrix.isApprox(scaler_static_matrix.get_variance()) )
    return 1;

  std::vector<Eigen::Matrix2f> expected_data_static_matrix({
    Eigen::Matrix2f({{-0.432246, -0.646559},
                     {-1.442280,  1.038707}}),
    Eigen::Matrix2f({{ 0.909395, -1.222748},
                     { 1.379051, -1.156408}}),
    Eigen::Matrix2f({{ 0.959824,  1.346780},
                     { 0.121229,  0.946792}}),
    Eigen::Matrix2f({{-1.436973,  0.522523},
                     {-0.057987, -0.829083}})
  });
  auto scaled_dataset_static_matrix(scaler_static_matrix.transform(*dataset_ptr_static_matrix));
  std::cout << "Scaled Dataset Static Matrix:" << std::endl;
  for(std::size_t i = 0; i < scaled_dataset_static_matrix.size(); ++i) {
    std::cout << scaled_dataset_static_matrix.data_at(i) << std::endl;
    if( !expected_data_static_matrix.at(i).isApprox(scaled_dataset_static_matrix.data_at(i)))
      return 1;
  }

  for(std::size_t i = 0; i < scaled_dataset_static_matrix.size(); ++i)
    if( !data_static_matrix.at(i).isApprox(scaler_static_matrix.inverse_transform(scaled_dataset_static_matrix.data_at(i))))
      return 1;

  // testing fit(dataset, fit_indices)
  std::vector<Eigen::VectorXf> data_specific_indices({
    (Eigen::VectorXf(7) << 0,        0,        0,        0,        0,        0,        0).finished(),
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
    (Eigen::VectorXf(7) << 0,        0,        0,        0,        0,        0,        0).finished(),
  });

  std::vector<int> labels_specific_indices({
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

  });

  auto dataset_ptr_specific_indices = std::make_shared< tudat_learn::Dataset<Eigen::VectorXf, int> >(tudat_learn::Dataset(data_specific_indices, labels_specific_indices));

  scaler_eigen.fit(*dataset_ptr_specific_indices, std::vector<size_t>({1,2,3,4,5,6,7,8,9,10}));

  std::cout << "Mean Specific Indices:\n" << scaler_eigen.get_mean().transpose() << std::endl;
  if( !expected_mean.isApprox(scaler_eigen.get_mean()) )
    return 1;

  std::cout << "Standard Deviation Specific Indices:\n" << scaler_eigen.get_standard_deviation().transpose() << std::endl;
  if( !expected_std.isApprox(scaler_eigen.get_standard_deviation()) )
    return 1;

  std::cout << "Variance Specific Indices:\n" << scaler_eigen.get_variance().transpose() << std::endl;
  if( !expected_variance.isApprox(scaler_eigen.get_variance()) )
    return 1;

  auto scaled_dataset_specific_indices(scaler_eigen.transform(*dataset_ptr_specific_indices, std::vector<size_t>({1,2,3,4,5,6,7,8,9,10})));
  std::cout << "Scaled Dataset Specific Indices:" << std::endl;
  for(std::size_t i = 0; i < scaled_dataset_specific_indices.size(); ++i) {
    std::cout << scaled_dataset_specific_indices.data_at(i).transpose() << std::endl;
    if( !expected_scaled_data.at(i).isApprox(scaled_dataset_specific_indices.data_at(i)))
      return 1;
  }

  for(std::size_t i = 0; i < scaled_dataset_specific_indices.size(); ++i)
    if( !data_dynamic_vector.at(i).isApprox(scaler_eigen.inverse_transform(scaled_dataset_specific_indices.data_at(i))))
      return 1;
  

  return 0;
}