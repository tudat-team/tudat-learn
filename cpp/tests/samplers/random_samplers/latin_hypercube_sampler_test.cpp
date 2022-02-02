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

  tudat_learn::LatinHypercubeSampler<Eigen::Vector2f> lhs(
    std::make_pair(Eigen::Vector2f(1,2), Eigen::Vector2f(3,4)),
    5
  );
  std::cout << lhs.test() << std::endl;

  tudat_learn::LatinHypercubeSampler<Eigen::Matrix2f> lhsm(
    std::make_pair(Eigen::Matrix2f({{1,2},{3,4}}), Eigen::Matrix2f({{3,4},{5,6}})),
    5
  );
  std::cout << lhsm.test() << std::endl;

  // for vectors!!!!s
  // tudat_learn::LatinHypercubeSampler<Eigen::Matrix2f> lhsm(
  //   std::make_pair(Eigen::Matrix2f({{1,2},{3,4}}), Eigen::Matrix2f({{3,4},{5,6}})),
  //   5
  // );
  // std::cout << lhsm.test() << std::endl;

  return 0;
}