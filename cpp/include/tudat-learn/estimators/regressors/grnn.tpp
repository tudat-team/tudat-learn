/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_GRNN_TPP
#define TUDAT_LEARN_GRNN_TPP

#ifndef TUDAT_LEARN_GRNN_HPP
#ERROR __FILE__ should only be included from rbfn.hpp
#endif

namespace tudat_learn
{

template <typename Datum_t, typename Label_t>
void GRNN<Datum_t, Label_t>::fit( ) {
  if(this->dataset_ptr->size() == 0) throw std::runtime_error("Dataset provided for fitting GRNN is empty. Please add some entries to the dataset.");

  this->center_points.resize(this->dataset_ptr->size(), this->dataset_ptr->data_at(0).rows());

  this->output_center_points.resize(this->dataset_ptr->size(), this->dataset_ptr->labels_at(0).rows());

  for(std::size_t i = 0; i < this->dataset_ptr->size(); ++i) {
    this->center_points.row(i) = this->dataset_ptr->data_at(i);
    this->output_center_points.row(i) = this->dataset_ptr->labels_at(i);
  }
}

template <typename Datum_t, typename Label_t>
void GRNN<Datum_t, Label_t>::fit(const std::vector<size_t> &fit_indices) {
  if(this->dataset_ptr->size() == 0) throw std::runtime_error("Dataset provided for fitting GRNN is empty. Please add some entries to the dataset.");
  if(fit_indices.size() == 0) throw std::runtime_error("fit_indices vector provided for fitting RBFN is empty. Please provide a non-empty vector.");

  this->center_points.resize(fit_indices.size(), this->dataset_ptr->data_at(0).rows());

  this->output_center_points.resize(fit_indices.size(), this->dataset_ptr->labels_at(0).rows());

  for(std::size_t i = 0; i < fit_indices.size(); ++i) {
    this->center_points.row(i) = this->dataset_ptr->data_at(fit_indices.at(i));
    this->output_center_points.row(i) = this->dataset_ptr->labels_at(fit_indices.at(i));
  }
}

template <typename Datum_t, typename Label_t>
Label_t GRNN<Datum_t, Label_t>::eval(const Datum_t &input) const {
  Eigen::Matrix<typename Datum_t::Scalar, 1, Eigen::Dynamic> rbf_output_vector(
    center_points.rows()
  );

  rbf_output_vector = rbf_ptr->eval_matrix(
      (center_points.rowwise() - input.transpose()).rowwise().norm()
  ).transpose();

  Label_t output_matrix(output_center_points.cols());
  
  output_matrix = (rbf_output_vector * output_center_points) / rbf_output_vector.sum();

  return output_matrix;  
} 

template <typename Datum_t, typename Label_t>
Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> GRNN<Datum_t, Label_t>::eval(const std::vector<Datum_t> &input_vector) const {
  using MatrixX = Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic>;

  MatrixX rbf_output_matrix(
    input_vector.size(), center_points.rows()
  );

  for(std::size_t i = 0; i < input_vector.size(); ++i) {
    rbf_output_matrix.row(i) = rbf_ptr->eval_matrix(
      (center_points.rowwise() - input_vector.at(i).transpose()).rowwise().norm()
    ).transpose(); 
  }

  MatrixX output_matrix(
    input_vector.size(), output_center_points.cols()
  );

  output_matrix = (rbf_output_matrix * output_center_points).array().colwise() / rbf_output_matrix.rowwise().sum().array();

  return output_matrix;
}

} // namespace tudat_learn

#endif // TUDAT_LEARN_GRNN_TPP