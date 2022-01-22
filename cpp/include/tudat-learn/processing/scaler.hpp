/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_SCALER_HPP
#define TUDAT_LEARN_SCALER_HPP

#include <vector>
#include <type_traits>

#include <Eigen/Core>

#include "tudat-learn/dataset.hpp"
#include "tudat-learn/processing.hpp"

namespace tudat_learn
{

template <typename Datum_t, typename Label_t>
class Scaler {
  public:  
    virtual void fit(const Dataset<Datum_t, Label_t> &dataset) = 0;

    virtual void fit(const Dataset<Datum_t, Label_t> &dataset, const std::vector<int> &fit_indices) = 0;

    virtual Dataset<Datum_t, Label_t> transform(Dataset<Datum_t, Label_t> dataset) const = 0;

    virtual Dataset<Datum_t, Label_t> transform(const Dataset<Datum_t, Label_t> &dataset, const std::vector<int> &fit_indices) const = 0;

    // virtual void fit_transform(Dataset<Datum_t, Label_t> &dataset) const = 0;

    // virtual Datum_t inverse_transform(Datum_t datum) const = 0;

  protected:
    // implements square root for the eigen type
    template <typename T>
    typename std::enable_if<          is_eigen<T>::value, 
    T>::type square_root(const T &arg) const { return Eigen::sqrt(arg.array()); }

    // implements square root for arithmetic types
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value, 
    T>::type square_root(const T &arg) const { return std::sqrt(arg); }

    // implements an element-wise multiplication for eigen types
    template <typename T>
    typename std::enable_if<           is_eigen<T>::value,
    T>::type operator_multiply_elementwise(const T &lhs, const T &rhs) const { return lhs.array() * rhs.array(); }

    // implements a multiplication for arithmetic types
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    T>::type operator_multiply_elementwise(const T &lhs, const T &rhs) const { return lhs * rhs; }

    // implements an element-wise division for eigen types
    template <typename T>
    typename std::enable_if<           is_eigen<T>::value,
    T>::type operator_divide_elementwise(const T &lhs, const T &rhs) const { return lhs.array() / rhs.array(); }

    // implements a division for arithmetic types
    template <typename T>
    typename std::enable_if< std::is_arithmetic<T>::value,
    T>::type operator_divide_elementwise(const T &lhs, const T &rhs) const { return lhs / rhs; }
};

} // namespace tudat_learn

#endif // TUDAT_LEARN_SCALER_HPP