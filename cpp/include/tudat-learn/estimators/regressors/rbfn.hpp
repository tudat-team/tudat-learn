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

    // template< typename Datum_tt = Datum_t, typename Label_tt = Label_t,
    //           std::enable_if< 
    template < typename Datum_tt = Datum_t, typename Label_tt = Label_t, 
               typename = std::enable_if_t< is_floating_point_eigen_matrix<Datum_tt>::value &&
                                            std::is_floating_point<Label_tt>::value 
                                          > 
    >
    RBFN(const std::shared_ptr< Dataset<Datum_tt, Label_tt> > &dataset_ptr,
         const std::shared_ptr< RBF<Label_t> > &rbf_ptr
    ) : 
    Regressor<Datum_tt, Label_tt>(dataset_ptr),
    rbf_ptr(rbf_ptr)
    { }

    void fit( ) override final;

  private:
    std::shared_ptr< RBF<Label_t> > rbf_ptr;
};
  
} // namespace tudat_learn

#include "tudat-learn/estimators/regressors/rbfn.tpp"

#endif // TUDAT_LEARN_RBFN_HPP
