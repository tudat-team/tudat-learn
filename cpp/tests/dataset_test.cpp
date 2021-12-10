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

#include <Eigen/Core>

#include "tudat-learn/dataset.hpp"

int test_get_closest_data( );

int main( ) {

  if(test_get_closest_data())
    return 1;

  return 0;
}

int test_get_closest_data( ) {
  std::vector< Eigen::VectorXf > data({
    (Eigen::VectorXf(2) <<  1.0f,  0.0f).finished(),
    (Eigen::VectorXf(2) <<  2.0f,  2.0f).finished(),
    (Eigen::VectorXf(2) <<  0.0f,  3.0f).finished(),
    (Eigen::VectorXf(2) << -4.0f,  4.0f).finished(),
    (Eigen::VectorXf(2) << -5.0f,  0.0f).finished(),
    (Eigen::VectorXf(2) << -6.0f, -6.0f).finished(),
    (Eigen::VectorXf(2) <<  0.0f, -7.0f).finished(),
    (Eigen::VectorXf(2) <<  8.0f, -8.0f).finished()
  });

  std::vector<float> labels({
    1, 2, 3, 4, 5, 6, 7, 8
  });

  tudat_learn::Dataset dataset(data, labels);

  {
    auto vector_of_interest = (Eigen::VectorXf(2) << 0.0f, 0.0f).finished();
    std::vector<int> expected({0, 1, 2, 3, 4, 5, 6, 7});
    std::vector<int> predicted = dataset.get_closest_data(vector_of_interest);

    std::cout << "Vector of interest:\n" << vector_of_interest << std::endl;

    std::cout << "Expected size is " << expected.size() << "." << std::endl;
    std::cout << "Predicted size is " << predicted.size() << "." << std::endl;

    if(expected.size() != predicted.size())
      return 1;

    for(int i = 0; i < expected.size(); ++i) {
      if(expected[i] != predicted[i])
        return 1;
      
      std::cout << "expected[i] = " << expected[i] << ", predicted[i] = " << predicted[i] << "." << std::endl;
    }

    {
      auto vector_of_interest = (Eigen::VectorXf(2) << 0.0f, 0.0f).finished();
      std::vector<int> expected({0, 1, 2, 3, 4, 5, 6, 7});
      std::vector<int> predicted = dataset.get_closest_data(vector_of_interest, -1);

      std::cout << "Vector of interest:\n" << vector_of_interest << std::endl;

      std::cout << "Expected size is " << expected.size() << "." << std::endl;
      std::cout << "Predicted size is " << predicted.size() << "." << std::endl;

      if(expected.size() != predicted.size())
        return 1;

      for(int i = 0; i < expected.size(); ++i) {
        if(expected[i] != predicted[i])
          return 1;
      
        std::cout << "expected[i] = " << expected[i] << ", predicted[i] = " << predicted[i] << "." << std::endl;
      }
    }

    {
      auto vector_of_interest = (Eigen::VectorXf(2) << 0.0f, 0.0f).finished();
      std::vector<int> expected({0, 1, 2, 4, 3, 6});
      std::vector<int> predicted = dataset.get_closest_data(vector_of_interest, 6);

      std::cout << "Vector of interest:\n" << vector_of_interest << std::endl;

      std::cout << "Expected size is " << expected.size() << "." << std::endl;
      std::cout << "Predicted size is " << predicted.size() << "." << std::endl;

      if(expected.size() != predicted.size())
        return 1;

      for(int i = 0; i < expected.size(); ++i) {
        if(expected[i] != predicted[i])
          return 1;
      
        std::cout << "expected[i] = " << expected[i] << ", predicted[i] = " << predicted[i] << "." << std::endl;
      }
    }

    {
      auto vector_of_interest = (Eigen::VectorXf(2) << -4.0f, 4.0f).finished();
      std::vector<int> expected({3, 2, 4, 1, 0, 5, 6});
      std::vector<int> expected_other({3, 4, 2, 1, 0, 5, 6});
      std::vector<int> predicted = dataset.get_closest_data(vector_of_interest, 7);

      std::cout << "Vector of interest:\n" << vector_of_interest << std::endl;

      std::cout << "Expected size is " << expected.size() << "." << std::endl;
      std::cout << "Predicted size is " << predicted.size() << "." << std::endl;

      if(expected.size() != predicted.size())
        return 1;

      for(int i = 0; i < expected.size(); ++i) {
        if(expected[i] != predicted[i] && expected_other[i] != predicted[i])
          return 1;
      
        std::cout << "expected[i] = " << expected[i] << " or " << expected_other[i] 
                  << ", predicted[i] = " << predicted[i] << "." << std::endl;
      }
    }

  }

  return 0;

}

