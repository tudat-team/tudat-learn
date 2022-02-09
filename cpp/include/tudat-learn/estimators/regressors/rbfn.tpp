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

// RBFN

template <typename Datum_t, typename Label_t>
void RBFN<Datum_t, Label_t>::fit( ) {
    using MatrixX = Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    if(this->dataset_ptr->size() == 0) throw std::runtime_error("Dataset provided for fitting RBFN is empty. Please add some entries to the dataset.");

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

    for(std::size_t i = 0; i < this->dataset_ptr->size(); ++i) {
        center_points.row(i)    = this->dataset_ptr->data_at(i);
        output_at_center.row(i) = this->dataset_ptr->labels_at(i);
    }

    // Creating the distance matrix
    // #rows and #columns = #(center points)
    MatrixX distance_matrix(
        this->dataset_ptr->size(), this->dataset_ptr->size()
    );

    for(std::size_t i = 0; i < this->dataset_ptr->size(); ++i) 
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

    if(this->dataset_ptr->size() == 0) throw std::runtime_error("Dataset provided for fitting RBFNPolynomial is empty. Please add some entries to the dataset.");
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

    for(std::size_t i = 0; i < fit_indices.size(); ++i) {
        center_points.row(i)    = this->dataset_ptr->data_at(fit_indices.at(i));
        output_at_center.row(i) = this->dataset_ptr->labels_at(fit_indices.at(i));
    }

    // Creating the distance matrix
    // #rows and #columns = #(center points)
    MatrixX distance_matrix(
        fit_indices.size(), fit_indices.size()
    );

    for(std::size_t i = 0; i < fit_indices.size(); ++i) 
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

    MatrixX distance_matrix_input(
        input_vector.size(), center_points.rows()
    );

    for(std::size_t i = 0; i < input_vector.size(); ++i) {
        distance_matrix_input.row(i) = rbf_ptr->eval_matrix(
            (center_points.rowwise() - input_vector.at(i).transpose()).rowwise().norm().transpose()
        );
    }

    MatrixX output_matrix(
        input_vector.size(), coefficients.cols()
    );

    output_matrix = distance_matrix_input * coefficients;

    return output_matrix;
}

template <typename Datum_t, typename Label_t>
Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> RBFN<Datum_t, Label_t>::gradient(const Datum_t &x) const {
    using MatrixX = Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    MatrixX gradient(
        this->coefficients.cols(), this->center_points.cols()
    );  
    
    gradient = this->coefficients.transpose() * this->rbf_ptr->gradient_rbfn(x, this->center_points);
    
    return gradient;
}

template <typename Datum_t, typename Label_t>
std::vector< Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> > RBFN<Datum_t, Label_t>::hessians(const Datum_t &x) const {
    using MatrixX = Eigen::Matrix< typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    
    std::vector<MatrixX> rbf_second_order_derivatives = this->rbf_ptr->hessian_rbfn(x, this->center_points);
    
    std::vector<MatrixX> hessians;
    hessians.reserve(this->coefficients.cols());
    
    for(int i = 0; i < this->coefficients.cols(); ++i) {
        MatrixX hessian(
            this->center_points.cols(), this->center_points.cols()
        );
    
        for(int j = 0; j < this->center_points.cols(); ++j) 
            hessian.row(j) = coefficients.col(i).transpose() * rbf_second_order_derivatives.at(j);
    
        hessians.push_back(hessian);
    }
    
    return hessians;
}

// RBFNPolynomial
template <typename Datum_t, typename Label_t>
void RBFNPolynomial<Datum_t, Label_t>::fit( ) {
    using MatrixX = Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    if(this->dataset_ptr->size() == 0) throw std::runtime_error("Dataset provided for fitting RBFNPolynomial is empty. Please add some entries to the dataset.");

    int n_cp = this->dataset_ptr->size(); // number of center points
    int in_dim = this->dataset_ptr->data_at(0).rows(); // input dimension
    int out_dim = this->dataset_ptr->labels_at(0).rows(); // output dimension

    // center_points matrix: has each center point as a column in the matrix 
    // Reads size of data vectors at runtime in case they are dynamic size matrices.
    // Assumes all the data vectors in the std::vector have the same size 
    // #rows = #(center points); #columns = input dimension
    this->center_points.resize(n_cp, in_dim);

    // Creating a matrix for the known values of the function at the center points
    // #rows = #(center points); #columns = output dimension
    MatrixX output_at_center(
        n_cp + in_dim + 1, 
        out_dim
    );

    for(int i = 0; i < n_cp; ++i) {
        this->center_points.row(i)    = this->dataset_ptr->data_at(i);
        output_at_center.row(i) = this->dataset_ptr->labels_at(i);
    }

    output_at_center.block(n_cp, 0, in_dim + 1, out_dim) = MatrixX::Zero(in_dim + 1, out_dim);

    // Creating the distance matrix
    // #rows and #columns = #(center points)
    MatrixX distance_matrix(
        n_cp, n_cp
    );

    for(int i = 0; i < n_cp; ++i) 
        distance_matrix.col(i) = (this->center_points.rowwise() - this->center_points.row(i)).rowwise().norm();

    distance_matrix = this->rbf_ptr->eval_matrix(distance_matrix);

    MatrixX matrix_to_invert(
        n_cp + in_dim + 1,
        n_cp + in_dim + 1
    );

    matrix_to_invert.block(       0,        0,       n_cp,       n_cp) = distance_matrix;
    matrix_to_invert.block(       0,     n_cp,       n_cp,          1) = MatrixX::Ones(n_cp, 1);
    matrix_to_invert.block(       0, 1 + n_cp,       n_cp,     in_dim) = this->center_points;
    matrix_to_invert.block(    n_cp,        0,          1,       n_cp) = MatrixX::Ones(1, n_cp);
    matrix_to_invert.block(1 + n_cp,        0,     in_dim,       n_cp) = this->center_points.transpose();
    matrix_to_invert.block(    n_cp,     n_cp, 1 + in_dim, 1 + in_dim) = MatrixX::Zero(1 + in_dim, 1 + in_dim);

    // resizing the coefficients vector:
    // #rows = #(center points); #columns = output dimension
    this->coefficients.resize(
        n_cp + in_dim + 1, 
        out_dim
    );

    for(int i = 0; i < out_dim; ++i) 
        this->coefficients.col(i) = matrix_to_invert.householderQr().solve(output_at_center.col(i));
}

template <typename Datum_t, typename Label_t>
void RBFNPolynomial<Datum_t, Label_t>::fit(const std::vector<int> &fit_indices) {
    using MatrixX = Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    if(this->dataset_ptr->size() == 0) throw std::runtime_error("Dataset provided for fitting RBFNPolynomial is empty. Please add some entries to the dataset.");
    if(fit_indices.size() == 0) throw std::runtime_error("fit_indices vector provided for fitting RBFNPolynomial is empty. Please provide a non-empty vector.");

    int n_cp = fit_indices.size(); // number of center points
    int in_dim = this->dataset_ptr->data_at(0).rows(); // input dimension
    int out_dim = this->dataset_ptr->labels_at(0).rows(); // output dimension

    // center_points matrix: has each center point as a column in the matrix 
    // Reads size of data vectors at runtime in case they are dynamic size matrices.
    // Assumes all the data vectors in the std::vector have the same size 
    // #rows = #(center points); #columns = input dimension
    this->center_points.resize(n_cp, in_dim);

    // Creating a matrix for the known values of the function at the center points
    // #rows = #(center points); #columns = output dimension
    MatrixX output_at_center(
        n_cp + in_dim + 1, 
        out_dim
    );

    for(int i = 0; i < n_cp; ++i) {
        this->center_points.row(i)    = this->dataset_ptr->data_at(fit_indices.at(i));
        output_at_center.row(i) = this->dataset_ptr->labels_at(fit_indices.at(i));
    }

    output_at_center.block(n_cp, 0, in_dim + 1, out_dim) = MatrixX::Zero(in_dim + 1, out_dim);

    // Creating the distance matrix
    // #rows and #columns = #(center points)
    MatrixX distance_matrix(
        n_cp, n_cp
    );

    for(int i = 0; i < n_cp; ++i) 
        distance_matrix.col(i) = (this->center_points.rowwise() - this->center_points.row(i)).rowwise().norm();

    distance_matrix = this->rbf_ptr->eval_matrix(distance_matrix);

    MatrixX matrix_to_invert(
        n_cp + in_dim + 1,
        n_cp + in_dim + 1
    );

    matrix_to_invert.block(       0,        0,       n_cp,       n_cp) = distance_matrix;
    matrix_to_invert.block(       0,     n_cp,       n_cp,          1) = MatrixX::Ones(n_cp, 1);
    matrix_to_invert.block(       0, 1 + n_cp,       n_cp,     in_dim) = this->center_points;
    matrix_to_invert.block(    n_cp,        0,          1,       n_cp) = MatrixX::Ones(1, n_cp);
    matrix_to_invert.block(1 + n_cp,        0,     in_dim,       n_cp) = this->center_points.transpose();
    matrix_to_invert.block(    n_cp,     n_cp, 1 + in_dim, 1 + in_dim) = MatrixX::Zero(1 + in_dim, 1 + in_dim);

    // resizing the coefficients vector:
    // #rows = #(center points); #columns = output dimension
    this->coefficients.resize(
        n_cp + in_dim + 1, 
        out_dim
    );

    for(int i = 0; i < out_dim; ++i) 
        this->coefficients.col(i) = matrix_to_invert.householderQr().solve(output_at_center.col(i));
}

template <typename Datum_t, typename Label_t>
Label_t RBFNPolynomial<Datum_t, Label_t>::eval(const Datum_t &input) const {
    using MatrixX = Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    // matrix that will contain the radial basis function values of the distances between the input and the center points
    MatrixX distance_matrix_input(
        1, this->coefficients.rows() // 1 is due to a single vector being processed in this function
    );

    distance_matrix_input = this->rbf_ptr->eval_matrix(
        (this->center_points.rowwise() - input.transpose()).rowwise().norm().transpose()
    );

    MatrixX concatenated_matrix(
        1, this->coefficients.rows()
    );
    concatenated_matrix << distance_matrix_input, Eigen::Matrix<typename Datum_t::Scalar, 1, 1>::Ones(), input.transpose();

    Label_t output_matrix(this->coefficients.cols());

    output_matrix = (concatenated_matrix * this->coefficients).transpose(); // transposed so that it returns a column vector

    return output_matrix;
}
  
template <typename Datum_t, typename Label_t>
Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> RBFNPolynomial<Datum_t, Label_t>::eval(const std::vector<Datum_t> &input_vector) const {
    using MatrixX = Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic>;

    MatrixX input_eigen(
        input_vector.size(), input_vector.at(0).rows() // 1 is due to a single vector being processed in this function
    );

    for(std::size_t i = 0; i < input_vector.size(); ++i) 
        input_eigen.row(i) = input_vector.at(i);

    MatrixX distance_matrix_input(
        input_vector.size(), this->center_points.rows()
    );

    for(std::size_t i = 0; i < input_vector.size(); ++i) {
        distance_matrix_input.row(i) = this->rbf_ptr->eval_matrix(
            (this->center_points.rowwise() - input_eigen.row(i)).rowwise().norm().transpose()
        );
    }

    MatrixX concatenated_matrix(
        input_vector.size(), this->coefficients.rows()
    );
    concatenated_matrix << distance_matrix_input, MatrixX::Ones(input_vector.size(), 1), input_eigen;

    MatrixX output_matrix(
        input_vector.size(), this->coefficients.cols()
    );

    output_matrix = concatenated_matrix * this->coefficients;

    return output_matrix;
}

template <typename Datum_t, typename Label_t>
Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> RBFNPolynomial<Datum_t, Label_t>::gradient(const Datum_t &x) const {
    using MatrixX = Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic>;    
    
    // Matrix that contains the partial derivatives dependant on the RBF that when multiplied by the
    // coefficients yield the gradient of the RBFN.
    // The bottom part of the matrix is of size [(dimension_input + 1) x dimension_input] and is of 
    // the form presented below:
    // 0 0 ... 0
    // 1 1 ... 1
    // 1 1 ... 1
    //   ...
    // 1 1 ... 1
    MatrixX partial_derivatives_rbf(
        this->center_points.rows() + this->center_points.cols() + 1, this->center_points.cols()
    );

    partial_derivatives_rbf << this->rbf_ptr->gradient_rbfn(x, this->center_points),
                               MatrixX::Zero(1, this->center_points.cols()),
                               MatrixX::Identity(this->center_points.cols(), this->center_points.cols());

    MatrixX gradient(
        this->coefficients.cols(), this->center_points.cols()
    );

    gradient = this->coefficients.transpose() * partial_derivatives_rbf;

    return gradient;
}

template <typename Datum_t, typename Label_t>
std::vector< Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> > RBFNPolynomial<Datum_t, Label_t>::hessians(const Datum_t &x) const {
    using MatrixX = Eigen::Matrix< typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    
    std::vector<MatrixX> rbf_second_order_derivatives = this->rbf_ptr->hessian_rbfn(x, this->center_points);
    
    std::vector<MatrixX> hessians;
    hessians.reserve(this->coefficients.cols());
    
    for(int i = 0; i < this->coefficients.cols(); ++i) {
        MatrixX hessian(
            this->center_points.cols(), this->center_points.cols()
        );
    
        for(int j = 0; j < this->center_points.cols(); ++j) 
            hessian.row(j) = this->coefficients.col(i).head(this->center_points.rows()).transpose() * rbf_second_order_derivatives.at(j);
    
        hessians.push_back(hessian);
    }
    
    return hessians;
}

} // namespace tudat_learn

#endif // TUDAT_LEARN_RBFN_TPP