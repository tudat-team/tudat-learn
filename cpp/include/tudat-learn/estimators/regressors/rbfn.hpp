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

#include "tudat-learn/dataset.hpp"
#include "tudat-learn/estimators/regressor.hpp"

namespace tudat_learn
{
  
class RBFN : public Regressor {
  public:
    RBFN();

    // RBFN(rbf, dimension of input/output, const RBF &rbf)

    void fit(const Dataset &dataset) override final{
        
    }
};
  
} // namespace tudat_learn


#endif // TUDAT_LEARN_RBFN_HPP
