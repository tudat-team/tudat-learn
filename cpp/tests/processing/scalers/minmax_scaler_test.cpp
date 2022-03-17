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

#include <tudat-learn/processing/scalers/minmax_scaler.hpp>

int main() {
  // Values generated using /tudat-learn/cpp/tests/python_scripts/processing/scalers/minmax_scaler_test.py
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

  tudat_learn::MinMaxScaler<Eigen::VectorXf, int> scaler_eigen;
  scaler_eigen.fit(*dataset_ptr);

  auto expected_min = (Eigen::VectorXf(7) << 0.071036, 0.060225, 0.020218, 0.118274, 0.097101, 0.110375, 0.018790).finished();
  std::cout << "Min:\n" << scaler_eigen.get_min().transpose() << std::endl;
  if( !expected_min.isApprox(scaler_eigen.get_min()) )
    return 1;

  auto expected_max = (Eigen::VectorXf(7) << 0.891773, 0.963663, 0.780529, 0.988374, 0.778157, 0.870012, 0.978618).finished();
  std::cout << "Max:\n" << scaler_eigen.get_max().transpose() << std::endl;
  if( !expected_max.isApprox(scaler_eigen.get_max()) )
    return 1;

  std::vector<Eigen::VectorXf> expected_scaled_data{
    (Eigen::VectorXf(7) << 0.582133, 0.724968, 0.766193, 0.490299, 0.479482, 0.704967, 0.436325).finished(),
    (Eigen::VectorXf(7) << 1.000000, 1.000000, 0.477731, 0.773993, 0.634007, 0.602485, 0.944760).finished(),
    (Eigen::VectorXf(7) << 0.000000, 0.029780, 0.000000, 0.820993, 1.000000, 1.000000, 1.000000).finished(),
    (Eigen::VectorXf(7) << 0.887158, 0.444141, 1.000000, 0.000000, 0.797027, 0.043413, 0.964630).finished(),
    (Eigen::VectorXf(7) << 0.549277, 0.392320, 0.321366, 0.753890, 0.527195, 0.602997, 0.000000).finished(),
    (Eigen::VectorXf(7) << 0.665986, 0.610857, 0.784831, 0.948712, 0.858548, 0.327963, 0.435747).finished(),
    (Eigen::VectorXf(7) << 0.763454, 0.000000, 0.850374, 0.634828, 0.166333, 0.024421, 0.309053).finished(),
    (Eigen::VectorXf(7) << 0.356600, 0.564479, 0.550280, 1.000000, 0.007259, 0.129670, 0.148485).finished(),
    (Eigen::VectorXf(7) << 0.709206, 0.213703, 0.586724, 0.144986, 0.090843, 0.000000, 0.664223).finished(),
    (Eigen::VectorXf(7) << 0.081813, 0.150931, 0.458374, 0.807630, 0.000000, 0.957786, 0.080544).finished(),
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

  tudat_learn::MinMaxScaler<Array7f, int> scaler_static_array;
  scaler_static_array.fit(*dataset_ptr_static_array);

  auto expected_min_static_array = Array7f(0.071036, 0.060225, 0.020218, 0.118274, 0.097101, 0.110375, 0.018790);
  std::cout << "Min Static Array:\n" << scaler_static_array.get_min().transpose() << std::endl;
  if( !expected_min_static_array.isApprox(scaler_static_array.get_min()) )
    return 1;

  auto expected_max_static_array = Array7f(0.891773, 0.963663, 0.780529, 0.988374, 0.778157, 0.870012, 0.978618);
  std::cout << "Max Static Array:\n" << scaler_static_array.get_max().transpose() << std::endl;
  if( !expected_max_static_array.isApprox(scaler_static_array.get_max()) )
    return 1;

  std::vector<Array7f> expected_scaled_data_static_array{
    Array7f(0.582133, 0.724968, 0.766193, 0.490299, 0.479482, 0.704967, 0.436325),
    Array7f(1.000000, 1.000000, 0.477731, 0.773993, 0.634007, 0.602485, 0.944760),
    Array7f(0.000000, 0.029780, 0.000000, 0.820993, 1.000000, 1.000000, 1.000000),
    Array7f(0.887158, 0.444141, 1.000000, 0.000000, 0.797027, 0.043413, 0.964630),
    Array7f(0.549277, 0.392320, 0.321366, 0.753890, 0.527195, 0.602997, 0.000000),
    Array7f(0.665986, 0.610857, 0.784831, 0.948712, 0.858548, 0.327963, 0.435747),
    Array7f(0.763454, 0.000000, 0.850374, 0.634828, 0.166333, 0.024421, 0.309053),
    Array7f(0.356600, 0.564479, 0.550280, 1.000000, 0.007259, 0.129670, 0.148485),
    Array7f(0.709206, 0.213703, 0.586724, 0.144986, 0.090843, 0.000000, 0.664223),
    Array7f(0.081813, 0.150931, 0.458374, 0.807630, 0.000000, 0.957786, 0.080544),
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

  tudat_learn::MinMaxScaler<float, int> scaler_scalar;
  scaler_scalar.fit(*dataset_ptr_scalar);

  float expected_min_scalar = 0.039188f;
  std::cout << "Min Scalar Array:\n" << scaler_scalar.get_min() << std::endl;
  if( std::abs(expected_min_scalar - scaler_scalar.get_min()) > 1e-6 )
    return 1;

  float expected_max_scalar = 0.976761f;
  std::cout << "Max Scalar Array:\n" << scaler_scalar.get_max() << std::endl;
  if( std::abs(expected_max_scalar - scaler_scalar.get_max()) > 1e-6 )
    return 1;

  std::vector<float> expected_scaled_data_scalar{
    0.999678, 0.458058, 1.000000, 0.603322, 0.746690, 0.000000, 0.259840, 0.086403, 0.274061, 0.084836
  };
  auto scaled_dataset_scalar(scaler_scalar.transform(*dataset_ptr_scalar));
  std::cout << "Scaled Dataset Scalar:" << std::endl;
  for(std::size_t i = 0; i < scaled_dataset_scalar.size(); ++i) {
    std::cout << scaled_dataset_scalar.data_at(i) << std::endl;
    if( std::abs(expected_scaled_data_scalar.at(i) - scaled_dataset_scalar.data_at(i)) > 1e-6 )
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

  tudat_learn::MinMaxScaler<Eigen::Matrix2f, int> scaler_static_matrix;
  scaler_static_matrix.fit(*dataset_ptr_static_matrix);

  auto expected_min_static_matrix = Eigen::Matrix2f({{0.131798, 0.265389},
                                                     {0.064147, 0.093941}});
  std::cout << "Min Static Matrix:\n" << scaler_static_matrix.get_min().transpose() << std::endl;
  if( !expected_min_static_matrix.isApprox(scaler_static_matrix.get_min()) )
    return 1;

  auto expected_max_static_matrix = Eigen::Matrix2f({{0.575946, 0.929296},
                                                     {0.523248, 0.692472}});
  std::cout << "Max Static Matrix:\n" << scaler_static_matrix.get_max().transpose() << std::endl;
  if( !expected_max_static_matrix.isApprox(scaler_static_matrix.get_max()) )
    return 1;

  std::vector<Eigen::Matrix2f> expected_data_static_matrix({
    Eigen::Matrix2f({{ 0.419196, 0.224239},
                     { 0.000000, 1.000000}}),
    Eigen::Matrix2f({{ 0.978960, 0.000000},
                     { 1.000000, 0.000000}}),
    Eigen::Matrix2f({{ 1.000000, 1.000000},
                     { 0.554174, 0.958127}}),
    Eigen::Matrix2f({{ 0.000000, 0.679219},
                     { 0.490652, 0.149115}})
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

  std::cout << "Min:\n" << scaler_eigen.get_min().transpose() << std::endl;
  if( !expected_min.isApprox(scaler_eigen.get_min()) )
    return 1;

  std::cout << "Max:\n" << scaler_eigen.get_max().transpose() << std::endl;
  if( !expected_max.isApprox(scaler_eigen.get_max()) )
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

  // testing a different range
  tudat_learn::MinMaxScaler<Eigen::VectorXf, int> scalar_range(std::pair<int, int>(-5, 7));
  scalar_range.fit(*dataset_ptr);

  auto expected_min_range = (Eigen::VectorXf(7) << 0.071036, 0.060225, 0.020218, 0.118274, 0.097101, 0.110375, 0.018790).finished();
  std::cout << "Min Range:\n" << scalar_range.get_min().transpose() << std::endl;
  if( !expected_min_range.isApprox(scalar_range.get_min()) )
    return 1;

  auto expected_max_range = (Eigen::VectorXf(7) << 0.891773, 0.963663, 0.780529, 0.988374, 0.778157, 0.870012, 0.978618).finished();
  std::cout << "Max Range:\n" << scalar_range.get_max().transpose() << std::endl;
  if( !expected_max_range.isApprox(scalar_range.get_max()) )
    return 1;

  std::vector<Eigen::VectorXf> expected_scaled_data_range{
    (Eigen::VectorXf(7) <<  1.985595,  3.699621,  4.194317,  0.883586,  0.753782,  3.459604,  0.235901).finished(),
    (Eigen::VectorXf(7) <<  7.000000,  7.000000,  0.732770,  4.287912,  2.608079,  2.229822,  6.337119).finished(),
    (Eigen::VectorXf(7) << -5.000000, -4.642645, -5.000000,  4.851916,  7.000000,  7.000000,  7.000000).finished(),
    (Eigen::VectorXf(7) <<  5.645890,  0.329694,  7.000000, -5.000000,  4.564324, -4.479046,  6.575561).finished(),
    (Eigen::VectorXf(7) <<  1.591325, -0.292157, -1.143610,  4.046684,  1.326334,  2.235967, -5.000000).finished(),
    (Eigen::VectorXf(7) <<  2.991827,  2.330278,  4.417978,  6.384540,  5.302571, -1.064441,  0.228962).finished(),
    (Eigen::VectorXf(7) <<  4.161449, -5.000000,  5.204493,  2.617938, -3.004006, -4.706950, -1.291361).finished(),
    (Eigen::VectorXf(7) << -0.720797,  1.773751,  1.603361,  7.000000, -4.912888, -3.443962, -3.218181).finished(),
    (Eigen::VectorXf(7) <<  3.510478, -2.435569,  2.040693, -3.260172, -3.909887, -5.000000,  2.970678).finished(),
    (Eigen::VectorXf(7) << -4.018243, -3.188825,  0.500491,  4.691562, -5.000000,  6.493437, -4.033477).finished(),
  };
  auto scaled_dataset_range(scalar_range.transform(*dataset_ptr));
  std::cout << "Scaled Dataset Range:" << std::endl;
  for(std::size_t i = 0; i < scaled_dataset_range.size(); ++i) {
    std::cout << scaled_dataset_range.data_at(i).transpose() << std::endl;
    if( !expected_scaled_data_range.at(i).isApprox(scaled_dataset_range.data_at(i)))
      return 1;
  }

  for(std::size_t i = 0; i < scaled_dataset_range.size(); ++i)
    if( !data_dynamic_vector.at(i).isApprox(scalar_range.inverse_transform(scaled_dataset_range.data_at(i))))
      return 1;


  return 0;
}