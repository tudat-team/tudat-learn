/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_CROSS_VALIDATION_HPP 
#define TUDAT_LEARN_CROSS_VALIDATION_HPP 

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "tudat-learn/dataset.hpp"
#include "tudat-learn/estimator.hpp"
#include "tudat-learn/model_evaluation/split.hpp"

namespace tudat_learn
{

template <typename Datum_t, typename Label_t>
class CrossValidation {
  public:
    /**
     * @brief Deleted default constructor, to ensure CrossValidation is created with settings
     * 
     */
    CrossValidation( ) = delete;

    CrossValidation(
      const std::shared_ptr< Dataset<Datum_t, Label_t> > &dataset_ptr,
      const std::shared_ptr< Estimator<Datum_t, Label_t> > &estimator_ptr,
      const std::shared_ptr< Split<Datum_t, Label_t> > &split_ptr,
      const std::vector< std::function<double(
        const std::shared_ptr<Dataset<Datum_t, Label_t>> &,
        const std::shared_ptr<Estimator<Datum_t, Label_t>> &, 
        const std::vector<size_t> &
      )> > &metrics
    ) :
    dataset_ptr(dataset_ptr),
    estimator_ptr(estimator_ptr),
    split_ptr(split_ptr),
    metrics(metrics)
    { }

    /**
     * @brief Function that performs cross-validation on the estimator.
     * 
     * @return std::vector< std::vector<double> > Vector of vector<double>. The number of vector<double> corresponds to the number
     * of splits, while the number of doubles in each of those vectors corresponds to the number of metrics.
     */
    std::vector< std::vector<double> > cross_validate( ) const;

  protected:
    std::shared_ptr< Dataset<Datum_t, Label_t> > dataset_ptr;

    std::shared_ptr< Estimator<Datum_t, Label_t> > estimator_ptr;

    std::shared_ptr< Split<Datum_t, Label_t> > split_ptr;

    std::vector< std::function<double(
      const std::shared_ptr<Dataset<Datum_t, Label_t>> &,
      const std::shared_ptr<Estimator<Datum_t, Label_t>> &, 
      const std::vector<size_t> &
    )> > metrics;
};

} // namespace tudat_learn


#include "tudat-learn/model_evaluation/cross_validation.tpp"

#endif // TUDAT_LEARN_CROSS_VALIDATION_HPP 