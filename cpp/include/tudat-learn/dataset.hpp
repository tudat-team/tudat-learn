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
    : data(data), labels(labels) { 
      if(data.size() != labels.size()) throw std::length_error("Vectors with Data and Targets must have the same length.\n");
    }

    Dataset(const std::vector<Datum_t> &data, const std::vector<Label_t> &labels, const std::vector<Label_t> &response)
    : data(data), labels(labels), response(response) { 
      if(data.size() != response.size()) throw std::length_error("Vectors with Data and Response must have the same length.\n");
    }

    const size_t size( ) const { return data.size(); }

          Datum_t &data_at(size_t pos)       { return data.at(pos); }
    const Datum_t &data_at(size_t pos) const { return data.at(pos); }

          Datum_t &labels_at(size_t pos)       { return labels.at(pos); }
    const Datum_t &labels_at(size_t pos) const { return labels.at(pos); }

    void push_back(const Datum_t  &datum, const Label_t  &label) {
      data.push_back(datum);
      labels.push_back(label);
    }
    
    void push_back(     Datum_t &&datum,        Label_t &&label) {
      data.push_back(std::move(datum));
      labels.push_back(std::move(label));
    }

    template <typename Datum_tt, typename Response_tt>
    friend Dataset<Datum_tt, Response_tt> get_datset_with_response(Dataset<Datum_tt> &dataset, std::vector<Response_tt> &response);

  private:
    std::vector<Datum_t> data;

    std::vector<Label_t> labels;

    std::vector<Label_t> response;
};

template <typename Datum_t>
class Dataset<Datum_t, none_t> {
  public:
    Dataset( ) { }

    Dataset(const std::vector<Datum_t> &data)
    : data(data) { }

    const size_t size( ) const { return data.size(); }

          Datum_t &data_at(size_t pos)       { return data.at(pos); }
    const Datum_t &data_at(size_t pos) const { return data.at(pos); }

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
 * @brief Creates a labelled dataset from an unlabelled dataset and a Response object. Vector is created with empty labels!
 * 
 * @tparam Datum_tt 
 * @tparam Response_tt 
 * @param dataset 
 * @param response 
 * @return Dataset<Datum_tt, Response_tt> 
 */
template <typename Datum_tt, typename Response_tt>
Dataset<Datum_tt, Response_tt> get_datset_with_response(Dataset<Datum_tt> &dataset, std::vector<Response_tt> &responses) {
  return Dataset<Datum_tt, Response_tt>(dataset.data, std::vector<Response_tt>(), responses);
}



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
