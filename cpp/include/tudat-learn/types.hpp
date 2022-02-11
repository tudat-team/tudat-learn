/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_TYPES_HPP
#define TUDAT_LEARN_TYPES_HPP

#include <vector>
#include <type_traits>

#include <Eigen/Core>


namespace tudat_learn
{
  /// None type to be used in classes that work with unlabelled data, such as an unlabelled dataset.
  typedef struct None_t { } none_t;

  /// Helper type trait for is_floating_point_eigen_vector.
  template <typename T> 
  struct __is_floating_point_eigen_vector_helper                                                     : std::false_type { };
  
  template <int RowsAtCompileTime> 
  struct __is_floating_point_eigen_vector_helper< Eigen::Matrix<      float, RowsAtCompileTime, 1> > : std::true_type { };

  template <int RowsAtCompileTime> 
  struct __is_floating_point_eigen_vector_helper< Eigen::Matrix<     double, RowsAtCompileTime, 1> > : std::true_type { };

  template <int RowsAtCompileTime> 
  struct __is_floating_point_eigen_vector_helper< Eigen::Matrix<long double, RowsAtCompileTime, 1> > : std::true_type { };

  /// Type trait for Eigen::Vector with floating-point type.
  template <typename F>
  struct is_floating_point_eigen_vector : __is_floating_point_eigen_vector_helper<typename std::remove_cv< F >::type>::type { };

  /// Type trait for either Eigen::Matrix or Eigen::Array types.
  template <typename T> 
  struct is_eigen                                                            : std::false_type { };

  template <int RowsAtCompileTime, int ColsAtCompileTime, typename T>
  struct is_eigen< Eigen::Matrix< T, RowsAtCompileTime, ColsAtCompileTime> > : std::true_type  { };

  template <int RowsAtCompileTime, int ColsAtCompileTime, typename T>
  struct is_eigen< Eigen::Array < T, RowsAtCompileTime, ColsAtCompileTime> > : std::true_type  { };

  /// Type trait for std::vector class.
  template <typename T>
  struct is_stl_vector                          : std::false_type { };

  template <typename ... Args>
  struct is_stl_vector< std::vector< Args...> > : std::true_type { };

  /**
 * @brief Constexpr to evaluate if a type is a floating_point Eigen type. \n 
 * Used when scalars may also be compared, as well as std::vectors.
 * 
 * @tparam T Type.
 * @return true If T is a floating-point Eigen type
 * @return false Otherwise.
 */
template <typename T>
constexpr bool is_floating_point_eigen(){
  if constexpr(is_eigen<T>::value)
    return std::is_floating_point<typename T::Scalar>::value;
  else
    return false;
}

/**
 * @brief Constexpr to evaluate if a type is a floating_point std::vector type. \n 
 * Used when scalars may also be compared, as well as Eigen types.
 * 
 * @tparam T Type.
 * @return true If T is a floating-point std::vector type
 * @return false Otherwise.
 */
template <typename T>
constexpr bool is_floating_point_stl_vector(){
  if constexpr(is_stl_vector<T>::value)
    return std::is_floating_point<typename T::value_type>::value;
  else
    return false;
}

} // namespace tudat_learn

#endif // TUDAT_LEARN_TYPES_HPP