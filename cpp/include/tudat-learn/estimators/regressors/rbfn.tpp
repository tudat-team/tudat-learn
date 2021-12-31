/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_RBFN_TPP
#define TUDAT_LEARN_RBFN_TPP

#ifndef TUDAT_LEARN_RBF_HPP
#ERROR __FILE__ should only be included from rbfn.hpp
#endif



namespace tudat_learn
{

template <typename Datum_t, typename Label_t>
void RBFN<Datum_t, Label_t>::fit( ) {
    using MatrixX = Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    if(this->dataset_ptr->size() == 0) throw std::runtime_error("Dataset provided for fitting RBFN is empty. Please provide a non-empty dataset.");

    // center_points matrix: has each center point as a column in the matrix 
    // Reads size of data vectors at runtime in case they are dynamic size matrices.
    // Assumes all the data vectors in the std::vector have the same size 
    // #rows = #(center points); #columns = input dimension
    center_points.resize(this->dataset_ptr->size(), this->dataset_ptr->data_at(0).rows());

    // Creating a matrix for the known values of the function at the center points
    // #rows = #(center points); #columns = output dimension
    MatrixX output_at_center(
        this->dataset_ptr->size(), this->dataset_ptr->labels_at(0).rows()
    );

    for(int i = 0; i < this->dataset_ptr->size(); ++i) {
        center_points.row(i)    = this->dataset_ptr->data_at(i);
        output_at_center.row(i) = this->dataset_ptr->labels_at(i);
    }

    // Creating the distance matrix
    // #rows and #columns = #(center points)
    MatrixX distance_matrix(
        this->dataset_ptr->size(), this->dataset_ptr->size()
    );

    for(int i = 0; i < this->dataset_ptr->size(); ++i) 
        distance_matrix.col(i) = (center_points.rowwise() - center_points.row(i)).rowwise().norm();

    distance_matrix = rbf_ptr->eval_matrix(distance_matrix);

    // resizing the coefficients vector:
    // #rows = #(center points); #columns = output dimension
    coefficients.resize(this->dataset_ptr->size(), this->dataset_ptr->labels_at(0).rows());

    for(int i = 0; i < coefficients.cols(); ++i) 
        coefficients.col(i) = distance_matrix.householderQr().solve(output_at_center.col(i));
    
}

template <typename Datum_t, typename Label_t>
void RBFN<Datum_t, Label_t>::fit(const std::vector<int> &fit_indices) {
    using MatrixX = Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    if(fit_indices.size() == 0) throw std::runtime_error("fit_indices vector provided for fitting RBFN is empty. Please provide a non-empty vector.");
    // center_points matrix: has each center point as a column in the matrix 
    // Reads size of data vectors at runtime in case they are dynamic size matrices.
    // Assumes all the data vectors in the std::vector have the same size 
    // #rows = #(indices); #columns = input dimension
    center_points.resize(fit_indices.size(), this->dataset_ptr->data_at(0).rows());


    // Creating a matrix for the known values of the function at the center points
    // #rows = #(center points); #columns = output dimension
    MatrixX output_at_center(
        fit_indices.size(), this->dataset_ptr->labels_at(0).rows()
    );

    for(int i = 0; i < fit_indices.size(); ++i) {
        center_points.row(i)    = this->dataset_ptr->data_at(fit_indices.at(i));
        output_at_center.row(i) = this->dataset_ptr->labels_at(fit_indices.at(i));
    }

    // Creating the distance matrix
    // #rows and #columns = #(center points)
    MatrixX distance_matrix(
        fit_indices.size(), fit_indices.size()
    );

    for(int i = 0; i < fit_indices.size(); ++i) 
        distance_matrix.col(i) = (center_points.rowwise() - center_points.row(i)).rowwise().norm();

    distance_matrix = rbf_ptr->eval_matrix(distance_matrix);

    // resizing the coefficients vector:
    // #rows = #(center points); #columns = output dimension
    coefficients.resize(fit_indices.size(), this->dataset_ptr->labels_at(0).rows());

    for(int i = 0; i < coefficients.cols(); ++i)
        coefficients.col(i) = distance_matrix.householderQr().solve(output_at_center.col(i));
}

template <typename Datum_t, typename Label_t>
Label_t RBFN<Datum_t, Label_t>::eval(const Datum_t &input) const {
    using MatrixX = Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    // matrix that will contain the radial basis function values of the distances between the input and the center points
    MatrixX distance_matrix_input(
        1, coefficients.rows() // 1 is due to a single vector being processed in this function
    );

    distance_matrix_input = rbf_ptr->eval_matrix(
        (center_points.rowwise() - input.transpose()).rowwise().norm().transpose()
    );

    Label_t output_matrix(coefficients.cols());
    output_matrix = (distance_matrix_input * coefficients).transpose(); // transposed so that it returns a column vector

    return output_matrix;
}

template <typename Datum_t, typename Label_t>
Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> RBFN<Datum_t, Label_t>::eval(const std::vector<Datum_t> &input_vector) const {
    using MatrixX = Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    MatrixX input_eigen(
        input_vector.size(), input_vector.at(0).rows() // 1 is due to a single vector being processed in this function
    );

    for(int i = 0; i < input_vector.size(); ++i) 
        input_eigen.row(i) = input_vector.at(i);

    MatrixX distance_matrix_input(
        input_vector.size(), center_points.rows()
    );

    for(int i = 0; i < input_vector.size(); ++i) {
        distance_matrix_input.row(i) = rbf_ptr->eval_matrix(
            (center_points.rowwise() - input_eigen.row(i)).rowwise().norm().transpose()
        );
    }

    MatrixX output_matrix(
        input_vector.size(), coefficients.cols()
    );

    output_matrix = distance_matrix_input * coefficients;

    return output_matrix;
}
  
} // namespace tudat_learn

#endif // TUDAT_LEARN_RBFN_TPP