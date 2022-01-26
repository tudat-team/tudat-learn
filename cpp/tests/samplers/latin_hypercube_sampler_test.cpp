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
#include <utility>

#include <Eigen/Core>

#include "tudat-learn/samplers/latin_hypercube_sampler.hpp"

int main() {

  tudat_learn::LatinHypercubeSampler<Eigen::Vector2f> lhs(
    std::make_pair(Eigen::Vector2f(1,2), Eigen::Vector2f(3,4)),
    5
  );

  return 0;
}