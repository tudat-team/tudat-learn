/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_K_FOLD_SPLIT_HPP 
#define TUDAT_LEARN_K_FOLD_SPLIT_HPP 

#include <algorithm>
#include <iterator>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "tudat-learn/dataset.hpp"
#include "tudat-learn/model_evaluation/split.hpp"
#include "tudat-learn/random.hpp"

namespace tudat_learn
{

template <typename Datum_t, typename Label_t>
class KFoldSplit : public Split<Datum_t, Label_t> {
  public:
     
    KFoldSplit( ) = delete;

    KFoldSplit(
      const std::shared_ptr< Dataset<Datum_t, Label_t> > &dataset_ptr,
      const size_t n_folds,
      const bool shuffle=false
    ) :
    Split<Datum_t, Label_t>(dataset_ptr, shuffle),
    n_folds(n_folds)
    { }

    /**
     * @brief Virtual destructor, as the class has virtual functions.
     * 
     */
    virtual ~KFoldSplit( ) { }


    virtual std::vector<std::pair<std::vector<size_t>, std::vector<size_t>>> split( ) const override;

    

  protected:
    size_t n_folds;


};

} // namespace tudat_learn

#include "tudat-learn/model_evaluation/splits/k_fold_split.tpp"

#endif // TUDAT_LEARN_K_FOLD_SPLIT_HPP 