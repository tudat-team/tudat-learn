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
#include <utility>

#include <Eigen/Core>

#include "tudat-learn/random.hpp"
#include "tudat-learn/samplers/random_samplers/latin_hypercube_sampler.hpp"

int main() {
  std::cout << std::is_floating_point<Eigen::VectorXd::Scalar>::value << std::endl;

  for(int i = 0; i < 10; ++i) {
    std::uniform_int_distribution<int> dis(0,10);
    std::mt19937 rng(0);
    std::cout << dis(rng) << std::endl;
  }

  std::vector<int> v({0,1,2,3,4,5,6,7,8});
  Eigen::MatrixXd m = Eigen::MatrixXd::NullaryExpr(3,3,
    [&](Eigen::Index i, Eigen::Index j){ return v.at(i * 3 + j); });
  std::cout << m << std::endl;

  tudat_learn::Random::set_seed(5);

  tudat_learn::LatinHypercubeSampler<Eigen::Vector2f> lhs(
    std::make_pair(Eigen::Vector2f(1,2), Eigen::Vector2f(3,4)),
    5
  );
  std::cout << lhs.test() << std::endl;
  std::cout << "Test1:" << std::endl;
  lhs.sample();

  tudat_learn::LatinHypercubeSampler<Eigen::Matrix2f> lhsm(
    std::make_pair(Eigen::Matrix2f({{1,2},{3,4}}), Eigen::Matrix2f({{3,4},{5,6}})),
    5
  );
  std::cout << lhsm.test() << std::endl;
  std::cout << "Another Test:" << std::endl;
  auto intervals = lhsm.sample();
  std::cout << "Aaaand:" << std::endl;
  for(const auto &it: intervals)
    std::cout << it << "\n" << std::endl;


  // for vectors!!!!s
  tudat_learn::LatinHypercubeSampler<std::vector<float>> lhsv(
    std::make_pair(std::vector<float>({1,2}), std::vector<float>({3,5})), 
    5
  );
  std::cout << lhsv.test() << std::endl;
  std::cout << "Another Test:" << std::endl;
  auto intervals_vec = lhsv.sample();
  lhsv.sample();
  lhsv.sample();
  std::cout << "Aaaand:" << std::endl;
  // for(const auto &it: intervals_vec)
  //   std::cout << it << "\n" << std::endl;

  return 0;
}