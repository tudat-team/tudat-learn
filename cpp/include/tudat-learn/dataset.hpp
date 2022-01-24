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

#include <algorithm>
#include <numeric>
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

    // Dataset(Dataset<Datum_t, Label_t> &&other) {
    //   data = std::move(other.data);
    //   labels = std::move(other.labels);
    //   response = std::move(other.response);
    // }

    // Dataset(const Dataset<Datum_t, Label_t> &other) {
    //   data = other.data;
    //   labels = other.labels;
    //   response = other.response;
    // }

    // Dataset<Datum_t, Label_t>& operator=(Dataset<Datum_t, Label_t> &&other) {
    //   if(this != &other) {
    //     data = std::move(other.data);
    //     labels = std::move(other.labels);
    //     response = std::move(other.response);
    //   }

    //   return *this;
    // }

    // Dataset<Datum_t, Label_t>& operator=(const Dataset<Datum_t, Label_t> &other) {
    //   if(this != &other) {
    //     data = other.data;
    //     labels = other.labels;
    //     response = other.response;
    //   }

    //   return *this;
    // }


    const size_t size( ) const { return data.size(); }

          Datum_t &data_at(size_t pos)       { return data.at(pos); }
    const Datum_t &data_at(size_t pos) const { return data.at(pos); }

          Label_t &labels_at(size_t pos)       { return labels.at(pos); }
    const Label_t &labels_at(size_t pos) const { return labels.at(pos); }

    void push_back(const Datum_t  &datum, const Label_t  &label) {
      data.push_back(datum);
      labels.push_back(label);
    }
    
    void push_back(     Datum_t &&datum,        Label_t &&label) {
      data.push_back(std::move(datum));
      labels.push_back(std::move(label));
    }

    void reserve(size_t new_cap) {
      data.reserve(new_cap);
      labels.reserve(new_cap);
    }

    /**
     * @brief Gets the `amount` indices of the closest instances to vector_of_interest in the dataset. Indices are sorted
     * from closest to furthest away if and only if amount < data.size(). Otherwise they are in natural order:
     * 0, 1, ..., data.size()-1
     * 
     * @tparam Datum_tt Must be an Eigen::Vector of floating point type.
     * @param vector_of_interest vector to which the closest data points will be found.
     * @param amount Desired amount of instances in the dataset. Defaults to data.size() if no amount is given or if
     * it is larger than the amount of instances in the vector
     * @return std::enable_if< std::is_floating_point<F>::value && is_floating_point_eigen_vector<Datum_tt>::value,
     * std::vector<int> >::type 
     */
    template <typename Datum_tt = Datum_t>
    typename std::enable_if< is_floating_point_eigen_vector<Datum_tt>::value, std::vector<int> >::type
    get_closest_data(Datum_tt vector_of_interest, int amount=-1);

    template <typename Datum_tt, typename Response_tt>
    friend Dataset<Datum_tt, Response_tt> get_dataset_with_response(Dataset<Datum_tt> &dataset, std::vector<Response_tt> &response);

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

    void reserve(size_t new_cap) {
      data.reserve(new_cap);
    }

    /**
     * @brief Gets the `amount` indices of the closest instances to vector_of_interest in the dataset. Indices are sorted
     * from closest to furthest away if and only if amount < data.size(). Otherwise they are in natural order:
     * 0, 1, ..., data.size()-1
     * 
     * @tparam Datum_tt Must be an Eigen::Vector of floating point type.
     * @param vector_of_interest vector to which the closest data points will be found.
     * @param amount Desired amount of instances in the dataset. Defaults to data.size() if no amount is given or if
     * it is larger than the amount of instances in the vector
     * @return std::enable_if< std::is_floating_point<F>::value && is_floating_point_eigen_vector<Datum_tt>::value,
     * std::vector<int> >::type 
     */
    template <typename Datum_tt = Datum_t>
    typename std::enable_if< is_floating_point_eigen_vector<Datum_tt>::value, std::vector<int> >::type
    get_closest_data(Datum_tt vector_of_interest, int amount=-1);

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
Dataset<Datum_tt, Response_tt> get_dataset_with_response(Dataset<Datum_tt> &dataset, std::vector<Response_tt> &responses) {
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

#include "dataset.tpp"

#endif // TUDAT_LEARN_DATASET_HPP
