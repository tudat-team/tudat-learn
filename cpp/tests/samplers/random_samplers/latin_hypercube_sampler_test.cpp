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
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "tudat-learn/operator.hpp"
#include "tudat-learn/random.hpp"
#include "tudat-learn/samplers/random_samplers/latin_hypercube_sampler.hpp"

namespace tudat_learn {

template <typename T>
struct VerifyLHS : public Operator<T> {
  VerifyLHS( ) : Operator<T>( ) { }

  bool verify_lhs(const std::vector<T> &samples, const std::pair<T, T> &range) {
    this->range = range;
    
    n_samples = samples.size();

    if(!(n_samples > 0))
      return false;

    n_dimensions = this->get_dimensions(samples.at(0));

    std::vector<int> single_indices_vector(n_samples);
    std::iota(single_indices_vector.begin(), single_indices_vector.end(), 0);
    std::vector<std::vector<int>> all_possible_indices(n_dimensions, single_indices_vector);

    range_size = this->operator_difference(range.second, range.first);
    bucket_size = this->operator_scalar_division(range_size, n_samples);

    for(int s = 0; s < n_samples; ++s) {
      get_bucket_index_from_sample(samples.at(s), all_possible_indices);
    }

    for(int d = 0; d < n_dimensions; ++d) {
      for(int s = 0; s < n_samples; ++s) {
        if(all_possible_indices.at(d).at(s) != -1)
          return false;
      }
    }

    return true;
  }

  protected:
    template <typename U>
    typename std::enable_if< std::is_arithmetic<U>::value,
    void>::type get_bucket_index_from_sample(U sample, std::vector<std::vector<int>> &all_possible_indices) const { 
      for(int d = 0; d < all_possible_indices.size(); ++d) { // should only yield a single iteration
        sample = (sample - range.first) / bucket_size;
        all_possible_indices.at(d).at(sample) = -1;
      }
    }

    template <typename U>
    typename std::enable_if<           is_eigen<U>::value,
    void>::type get_bucket_index_from_sample(U sample, std::vector<std::vector<int>> &all_possible_indices) const { 
      sample = (sample - range.first).array() / bucket_size.array();
      for(int d = 0; d < all_possible_indices.size(); ++d) {
        int r = d / sample.cols();
        int c = d % sample.cols();
        all_possible_indices.at(d).at(sample.row(r).col(c).value()) = -1;
      }
    }

    template <typename U>
    typename std::enable_if< std::is_arithmetic<U>::value,
    void>::type get_bucket_index_from_sample(std::vector<U> sample, std::vector<std::vector<int>> &all_possible_indices) const {        
      for(int d = 0; d < all_possible_indices.size(); ++d) {
        sample.at(d) = sample.at(d) - range.first.at(d) / bucket_size.at(d);
        all_possible_indices.at(d).at(sample.at(d)) = -1;
      }
    }

  protected:
    int n_samples;
    int n_dimensions;
    std::pair<T,T> range;
    T range_size;
    T bucket_size;

};

} // namespace tudat_learn

int main() {

  tudat_learn::Random::set_seed(1);

  // Eigen Static Vector
  auto range_static_eigen_vector = std::make_pair(Eigen::Vector2f(0,0), Eigen::Vector2f(5,10));
  tudat_learn::LatinHypercubeSampler<Eigen::Vector2f> lhs_static_eigen_vector(range_static_eigen_vector, 5);
  auto samples_static_eigen_vector = lhs_static_eigen_vector.sample();
  std::cout << "\nSamples Static Eigen Vector:" << std::endl;
  for(const auto &it : samples_static_eigen_vector)
    std::cout << it.transpose() << std::endl;

  tudat_learn::VerifyLHS<Eigen::Vector2f> verify_static_eigen_vector;
  if(!verify_static_eigen_vector.verify_lhs(samples_static_eigen_vector,range_static_eigen_vector))
    return 1;

  // Eigen Dynamic Array
  auto range_dynamic_eigen_array = std::make_pair(
    (Eigen::ArrayXXf(3,3) << 1, 2, 3, 4,  5,  6,  7,  8,  9).finished(), 
    (Eigen::ArrayXXf(3,3) << 2, 4, 6, 8, 10, 12, 14, 16, 18).finished()
  );
  tudat_learn::LatinHypercubeSampler<Eigen::ArrayXXf> lhs_dynamic_eigen_array(range_dynamic_eigen_array, 10);
  auto samples_dynamic_eigen_array = lhs_dynamic_eigen_array.sample();
  std::cout << "\nSamples Static Eigen Vector:" << std::endl;
  for(const auto &it : samples_dynamic_eigen_array)
    std::cout << it << "\n" << std::endl;

  tudat_learn::VerifyLHS<Eigen::ArrayXXf> verify_dynamic_eigen_array;
  if(!verify_dynamic_eigen_array.verify_lhs(samples_dynamic_eigen_array,range_dynamic_eigen_array))
    return 1;

  // Scalar
  auto range_scalar = std::make_pair(1.0, 2.0);
  tudat_learn::LatinHypercubeSampler<double> lhs_scalar(range_scalar, 10);
  auto samples_scalar = lhs_scalar.sample();
  std::cout << "\nSamples Static Eigen Vector:" << std::endl;
  for(const auto &it : samples_scalar)
    std::cout << it << "\n" << std::endl;

  tudat_learn::VerifyLHS<double> verify_scalar;
  if(!verify_scalar.verify_lhs(samples_scalar,range_scalar))
    return 1;


  // tudat_learn::LatinHypercubeSampler<Eigen::Vector2f> lhs(
  //   std::make_pair(Eigen::Vector2f(0,0), Eigen::Vector2f(5,10)),
  //   5
  // );
  
  // std::cout << "Test1:" << std::endl;
  // auto samples = lhs.sample();
  // std::cout << "And now testing the samples, the result is: " << verify.verify_lhs(samples, std::make_pair(Eigen::Vector2f(0,0), Eigen::Vector2f(5,10))) << std::endl;
  // samples.at(0).row(0) = (Eigen::MatrixXf(1,1) << 0.5).finished();
  // std::cout << "And now testing the samples, the result is: " << verify.verify_lhs(samples, std::make_pair(Eigen::Vector2f(0,0), Eigen::Vector2f(5,10))) << std::endl;

  // tudat_learn::LatinHypercubeSampler<Eigen::Matrix2f> lhsm(
  //   std::make_pair(Eigen::Matrix2f({{1,2},{3,4}}), Eigen::Matrix2f({{3,4},{5,6}})),
  //   5
  // );
  
  // std::cout << "Another Test:" << std::endl;
  // auto intervals = lhsm.sample();
  // std::cout << "Aaaand:" << std::endl;
  // for(const auto &it: intervals)
  //   std::cout << it << "\n" << std::endl;


  // // for vectors!!!!s
  // tudat_learn::LatinHypercubeSampler<std::vector<float>> lhsv(
  //   std::make_pair(std::vector<float>({1,2}), std::vector<float>({3,5})), 
  //   5
  // );
  
  // std::cout << "Vector Test:" << std::endl;
  // auto intervals_vec = lhsv.sample();
  // std::cout << "Aaaand:" << std::endl;
  // // for(const auto &it: intervals_vec)
  // //   std::cout << it << "\n" << std::endl;

  return 0;
}