/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_CLASSIFIER_HPP
#define TUDAT_LEARN_CLASSIFIER_HPP

#include "tudat-learn/estimators/regressor.hpp"

namespace tudat_learn
{

template <typename Datum_t, typename Label_t = none_t>
class Classifier : public Estimator<Datum_t, Label_t> {
  public:
    Classifier(const std::shared_ptr< Dataset<Datum_t, Label_t> > &dataset_ptr)
    : Estimator<Datum_t, Label_t>(dataset_ptr) { }

    virtual void fit( ) = 0;

};
  
} // namespace tudat_learn


#endif // TUDAT_LEARN_CLASSIFIER_HPP
