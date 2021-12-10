/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_RBFN_HPP
#define TUDAT_LEARN_RBFN_HPP

#include <memory>
#include <type_traits>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "tudat-learn/dataset.hpp"
#include "tudat-learn/types.hpp"
#include "tudat-learn/estimators/regressor.hpp"
#include "tudat-learn/estimators/regressors/rbf.hpp"

namespace tudat_learn
{
  
template <typename Datum_t, typename Label_t>
class RBFN : public Regressor<Datum_t, Label_t> {
  public:

    /**
     * @brief Deleting the default constructor to make sure the object is created with settings.
     * 
     */
    RBFN() = delete;

    // Requires Label_tt and Datum_tt to be vectors of the same floating-point type
    template < typename Datum_tt = Datum_t, typename Label_tt = Label_t, 
               typename = std::enable_if_t< is_floating_point_eigen_vector<Datum_tt>::value &&
                                            is_floating_point_eigen_vector<Label_tt>::value &&
                                            std::is_same<typename Datum_tt::Scalar, typename Label_tt::Scalar>::value
                                          > 
    >
    RBFN(const std::shared_ptr< Dataset<Datum_tt, Label_tt> > &dataset_ptr,
         const std::shared_ptr< RBF<typename Label_tt::Scalar> > &rbf_ptr
    ) : 
    Regressor<Datum_tt, Label_tt>(dataset_ptr),
    rbf_ptr(rbf_ptr)
    { }

    virtual void fit( ) override final;

    void fit(const std::vector<int> &fit_indices);

    Label_t eval(const Datum_t &input) const;

    Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> eval(const std::vector<Datum_t> &input_vector) const;

    const Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> get_coefficients( ) const {
      return coefficients;
    }

  private:
    std::shared_ptr< RBF<typename Label_t::Scalar> > rbf_ptr;

    Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic > center_points;

    Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> coefficients; 
};
  
} // namespace tudat_learn

#include "tudat-learn/estimators/regressors/rbfn.tpp"

#endif // TUDAT_LEARN_RBFN_HPP
