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

/**
 * @brief Labelled Dataset. \n
 * The Label_t is set as none_t by default, which creates an unlabelled dataset when no Label_t is chosen.
 * 
 * @tparam Datum_t The type of a single feature vector. Can be a simple scalar, an std::vector<T>, an Eigen::Matrix, ...
 * @tparam Label_t The type of a single label. Like the Datum_t, can be a simple scalar, an std::vector<T>, an Eigen::Matrix, ...
 */
template <typename Datum_t, typename Label_t = none_t>
class Dataset {

  public:
    /**
     * @brief Default constructor for the Dataset class.
     * 
     */
    Dataset( ) = default;

    /**
     * @brief Constructor for the Dataset class. \n 
     * Takes an std::vector<Datum_t> and an std::vector<Label_t> as inputs, saving copies of them in the 
     * Dataset::data and Dataset::labels fields. The vectors must have the same size.
     * 
     * @param data Vector of Datum_t, that is, vector of feature vectors.
     * @param labels Vector of Label_t, that is, vector of labels.
     */
    Dataset(const std::vector<Datum_t> &data, const std::vector<Label_t> &labels)
    : data(data), labels(labels) { 
      if(data.size() != labels.size()) throw std::length_error("Vectors with Data and Targets must have the same length.\n");
    }

    /**
     * @brief Constructor for the Dataset class. \n
     * Takes an std::vector<Datum_t> and two std::vector<Label_t> as inputs: one for the labels, and one for the response. Copies
     * of those vectors are saved in the Dataset::data, Dataset::labels, and Dataset::responses fields.
     * The data vector and the response vectors must have the same size. \n
     * This constructor is used to create a Labelled dataset from an unlabelled dataset. However, it is still not certain if this
     * is useful, so it should be reviewed in the future.
     * 
     * @param data Vector of Datum_t, that is, vector of feature vectors.
     * @param labels Vector of Label_t, that is, vector of labels.
     * @param response Vector of Label_t, which is the vector of response values. Response is the predicted result, which is not
     * necessarily the same as the label. In the predicted use-case for this constructor, there wouldn't even be labels.
     */
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

    /**
     * @brief Returns the number of elements in the data vector in the dataset. 
     * 
     * @return const size_t The number of elements in the container.
     */
    const size_t size( ) const { return data.size(); }

    /**
     * @brief Returns a reference to the data element at the specified location pos, with bounds checking. Similar to
     * std containers' at() method. In fact, it uses that method.
     * 
     * @param pos Position of the element to return.
     * @return Datum_t& Reference to the requested element.
     */
          Datum_t &data_at(size_t pos)       { return data.at(pos); }

    /**
     * @brief Returns a constant reference to the data element at the specified location pos, with bounds checking. Similar to
     * std containers' at() method. In fact, it uses that method.
     * 
     * @param pos Position of the element to return.
     * @return Datum_t& Constant reference to the requested element.
     */
    const Datum_t &data_at(size_t pos) const { return data.at(pos); }

    /**
     * @brief Returns a reference to the labels element at the specified location pos, with bounds checking. Similar to
     * std containers' at() method. In fact, it uses that method.
     * 
     * @param pos Position of the element to return.
     * @return Label_t& Reference to the requested element.
     */
          Label_t &labels_at(size_t pos)       { return labels.at(pos); }

    /**
     * @brief Returns a constant reference to the labels element at the specified location pos, with bounds checking. Similar to
     * std containers' at() method. In fact, it uses that method.
     * 
     * @param pos Position of the element to return.
     * @return Label_t& Constant reference to the requested element.
     */
    const Label_t &labels_at(size_t pos) const { return labels.at(pos); }

    /**
     * @brief Appends the given elements datum and label at the end of the respective data and labels vectors. The new values
     * are initialized as copies of datum and label. Similar to std containers' push_back() method. In fact, it uses that method.
     * 
     * @param datum The value of the element to append to data.
     * @param label The value of the element to append to labels.
     */
    void push_back(const Datum_t  &datum, const Label_t  &label) {
      data.push_back(datum);
      labels.push_back(label);
    }
    
    /**
     * @brief Appends the given elements datum and label at the end of the respective data and labels vectors. The new values
     * are moved into the new elements at each of the vectors. Similar to std containers' push_back() method. In fact, it uses
     * that method.
     * 
     * @param datum The value of the element to append to data.
     * @param label The value of the element to append to labels.
     */
    void push_back(     Datum_t &&datum,        Label_t &&label) {
      data.push_back(std::move(datum));
      labels.push_back(std::move(label));
    }

    /**
     * @brief Increases the capacity of the data and labels vectors to a value that's greater or equal to new_cap. If the new_cap
     * is greater than the current capacity() of those vectors, new storage is allocated, otherwise the function does nothing. 
     * Similar to std containers' reserve() method. In fact, it uses that method.
     * 
     * @param new_cap New capacity of the vector.
     */
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
    typename std::enable_if< is_floating_point_eigen_vector<Datum_tt>::value, 
    std::vector<int> >::type get_closest_data(Datum_tt vector_of_interest, int amount=-1);

    template <typename Datum_tt, typename Response_tt>
    friend Dataset<Datum_tt, Response_tt> get_dataset_with_response(Dataset<Datum_tt> &dataset, std::vector<Response_tt> &response);

  private:
    std::vector<Datum_t> data;      /**< Vector with the feature vectors. */

    std::vector<Label_t> labels;    /**< Vector with the labels. */

    std::vector<Label_t> response;  /**< Vector with the responses. */
};

/**
 * @brief Unlabelled Dataset. \n 
 * It is a specialization of the Labelled Dataset for labels(Label_t) of none_t.
 * 
 * @tparam Datum_t The type of a single feature vector. Can be a simple scalar, an std::vector<T>, an Eigen::Matrix, ...
 */
template <typename Datum_t>
class Dataset<Datum_t, none_t> {
  public:

    /**
     * @brief Default constructor for the Dataset class.
     * 
     */
    Dataset( ) = default;

    /**
     * @brief Constructor for the Dataset class. \n
     * Takes an std::vector<Datum_t> and saves a copy of it in the Dataset::data field.
     * 
     * @param data Vector of Datum_t, that is, vector of feature vectors.
     */
    Dataset(const std::vector<Datum_t> &data)
    : data(data) { }

    /**
     * @brief Returns the number of elements in the data vector in the dataset. 
     * 
     * @return const size_t The number of elements in the container.
     */
    const size_t size( ) const { return data.size(); }

    /**
     * @brief Returns a reference to the data element at the specified location pos, with bounds checking. Similar to
     * std containers' at() method. In fact, it uses that method.
     * 
     * @param pos Position of the element to return.
     * @return Datum_t& Reference to the requested element.
     */
          Datum_t &data_at(size_t pos)       { return data.at(pos); }

    /**
     * @brief Returns a constant reference to the data element at the specified location pos, with bounds checking. Similar to
     * std containers' at() method. In fact, it uses that method.
     * 
     * @param pos Position of the element to return.
     * @return Datum_t& Constant reference to the requested element.
     */
    const Datum_t &data_at(size_t pos) const { return data.at(pos); }

    /**
     * @brief Appends the given element datum data vector. The new value is initialized as a copy of datum. 
     * Similar to std containers' push_back() method. In fact, it uses that method.
     * 
     * @param datum The value of the element to append to data.
     */
    void push_back(const Datum_t  &datum) {
        data.push_back(datum);
    }
    
    /**
     * @brief Appends the given element datum data vector. The new value is moved into the new element at the end of the vector. 
     * Similar to std containers' push_back() method. In fact, it uses that method.
     * 
     * @param datum The value of the element to append to data.
     */
    void push_back(     Datum_t &&datum) {
      data.push_back(std::move(datum));
    }

    /**
     * @brief Increases the capacity of the data vectors to a value that's greater or equal to new_cap. If the new_cap is
     * greater than the current capacity() of those vector, new storage is allocated, otherwise the function does nothing.
     * Similar to std containers' reserve() method. In fact, it uses that method.
     * 
     * @param new_cap New capacity of the vector.
     */
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
    std::vector<Datum_t> data; /**< Vector with the feature vectors. */
};


/**
 * @brief Creates a labelled dataset from an unlabelled dataset and a vector of responses. \n
 * The dataset is created with an empty response objects, the vector of responses and dataset.data must have the same
 * length.
 * 
 * 
 * @tparam Datum_t The type of a single feature vector. Can be a simple scalar, an std::vector<T>, an Eigen::Matrix, ...
 * @tparam Label_t The type of a single label. Like the Datum_t, can be a simple scalar, an std::vector<T>, an Eigen::Matrix, ...
 * @param dataset Unlabelled dataset.
 * @param responses Vector with resonses.
 * @return Dataset<Datum_tt, Response_tt> Labelled dataset with empty labels, and a filled vector of responses.
 */
template <typename Datum_tt, typename Response_tt>
Dataset<Datum_tt, Response_tt> get_dataset_with_response(const Dataset<Datum_tt> &dataset, std::vector<Response_tt> &responses) {
  return Dataset<Datum_tt, Response_tt>(dataset.data, std::vector<Response_tt>(), responses);
}


  
} // namespace tudat_learn

#include "tudat-learn/dataset.tpp"

#endif // TUDAT_LEARN_DATASET_HPP
