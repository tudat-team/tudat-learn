/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_GRNN_HPP
#define TUDAT_LEARN_GRNN_HPP

#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <Eigen/Core>

#include "tudat-learn/dataset.hpp"
#include "tudat-learn/types.hpp"
#include "tudat-learn/estimators/regressor.hpp"
#include "tudat-learn/estimators/regressors/rbf.hpp"

namespace tudat_learn
{

/**
 * @brief Implementation of Generalized Regression Neural Networks.
 * From: D. F. Specht, A general regression neural network.
 * 
 * @tparam Datum_t Has to be an Eigen::Vector (meaning: column vector) of a floating-point type. The
 * floating-point type must be the same as Label_t's.
 * @tparam Label_t Has to be an Eigen::Vector (meaning: column vector) of a floating-point type. The
 * * floating-point type must be the same as Datum_t's.
 */
template <typename Datum_t, typename Label_t>
class GRNN : public Regressor<Datum_t, Label_t> {

  public:
    /**
    * @brief Deleting the default constructor to make sure the object is created with settings.
    */
    GRNN() = delete;

    template <typename Datum_tt = Datum_t, typename Label_tt = Label_t,
              typename = std::enable_if_t< is_floating_point_eigen_vector<Datum_tt>::value &&
                                           is_floating_point_eigen_vector<Label_tt>::value &&
                                           std::is_same<typename Datum_tt::Scalar, typename Label_tt::Scalar>::value
                          >
    >
    GRNN(
      const std::shared_ptr< Dataset<Datum_tt, Label_tt> > &dataset_ptr,
      const std::shared_ptr< RBF<typename Label_tt::Scalar> > &rbf_ptr
    ) :
    Regressor<Datum_tt, Label_tt>(dataset_ptr),
    rbf_ptr(rbf_ptr)
    { }

    virtual void fit( ) override;

    virtual void fit(const std::vector<int> &fit_indices);

    virtual Label_t eval(const Datum_t &input) const override;

    virtual Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> eval(const std::vector<Datum_t> &input_vector) const;    

  protected:
    std::shared_ptr< RBF<typename Label_t::Scalar> > rbf_ptr;

    Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic > center_points;

    Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> output_center_points; 
};

} // namespace tudat_learn

#include "tudat-learn/estimators/regressors/grnn.tpp"

#endif // TUDAT_LEARN_GRNN_HPP