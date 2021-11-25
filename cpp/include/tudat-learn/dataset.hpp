/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_DATASET_HPP
#define TUDAT_LEARN_DATASET_HPP

#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "tudat-learn/types.hpp"

namespace tudat_learn
{

template <typename Datum_t, typename Label_t = none_t>
class Dataset {

  public:
    Dataset( ) { }

    Dataset(const std::vector<Datum_t> &data, const std::vector<Label_t> &labels)
    : data(data), labels(labels) { }

    void push_back(const Datum_t  &datum, const Label_t  &label) {
      data.push_back(datum);
      labels.push_back(label);
    }
    
    void push_back(     Datum_t &&datum,        Label_t &&label) {
      data.push_back(std::move(datum));
      labels.push_back(std::move(label));
    }

  private:
    std::vector<Datum_t> data;

    std::vector<Label_t> labels;
};

template <typename Datum_t>
class Dataset<Datum_t, none_t> {
  public:
    Dataset( ) { }

    Dataset(const std::vector<Datum_t> &data)
    : data(data) { }

    void push_back(const Datum_t  &datum) {
        data.push_back(datum);
      }
      
    void push_back(     Datum_t &&datum) {
      data.push_back(std::move(datum));
    }

  private:
    std::vector<Datum_t> data;
};

/**
 * @brief Commented other possible implementations of Dataset
 * 
 */

// class BaseDataset { };

// template <typename Datum_t>
// class UnlabelledDataset : public BaseDataset {
//   public:
//     UnlabelledDataset(const std::vector< Datum_t > &data) : data(data) { }

//   private:
//     std::vector< Datum_t > data;
// };

// template <typename Datum_t, typename Target_t>
// class LabelledDataset : public BaseDataset {
//   public:
//     LabelledDataset(const std::vector< Datum_t > &data, const std::vector< Target_t > &targets) 
//       : data(data), targets(targets) { 
//         if(data.size() != targets.size()) throw std::length_error("Vectors with Data and Targets must have the same length.\n");
//     }

//   private:
//     std::vector< Datum_t > data;
//     std::vector< Target_t > targets;
// };

// typedef class Null_t {

// } null_t;

// template< typename Datum_t, typename Target_t = null_t >
// class Dataset : public BaseDataset {
//   public:
//     Dataset(const std::vector< Datum_t > &data) : data(data) { }

//     Dataset(const std::vector< Datum_t > &data, const std::vector< Target_t > &targets) 
//     : data(data), targets(targets) { 
//       if(data.size() != targets.size()) throw std::length_error("Vectors with Data and Targets must have the same length.\n");
//     }

//     std::vector< Datum_t > data;

//     std::vector< Target_t > targets;

// };
  
// class Dataset {
  
//   dataset_t data;

//   public:
//     Dataset(const dataset_t &dataset);

//     /**
//      * @brief Access datum_t at position n in the vector of dataset_t.
//      * 
//      * @param n 
//      * @return datum_t& reference to the datum_t element
//      */
//     datum_t &at(const size_t n) {
//       return data.at(n);
//     }

//     /**
//      * @brief Access datum_t at position n in the vector of dataset_t.
//      * 
//      * @param n 
//      * @return const datum_t& constant reference to the datum_t element
//      */
//     const datum_t &at(const size_t n) const {
//       return data.at(n);
//     }
// };
  
} // namespace tudat_learn


#endif // TUDAT_LEARN_DATASET_HPP
