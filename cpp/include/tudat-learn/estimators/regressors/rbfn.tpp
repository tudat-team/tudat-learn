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
    using scalar = typename Datum_t::Scalar;

    // center_points matrix: has each center point as a column in the matrix 
    // Reads size of data vectors at runtime in case they are dynamic size matrices.
    // Assumes all the data vectors in the std::vector have the same size 
    // #rows = #(center points); #columns = input dimension
    Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic > center_points(
        this->dataset_ptr->size(), this->dataset_ptr->data_at(0).rows()
    );

    // Creating a matrix for the known values of the function at the center points
    // #rows = #(center points); #columns = output dimension
    Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic > output_at_center(
        this->dataset_ptr->size(), this->dataset_ptr->labels_at(0).rows()
    );

    for(int i = 0; i < this->dataset_ptr->size(); ++i) {
        center_points.row(i)    = this->dataset_ptr->data_at(i);
        output_at_center.row(i) = this->dataset_ptr->labels_at(i);
    }

    // Creating the distance matrix
    // #rows and #columns = #(center points)
    Eigen::Matrix<scalar, Eigen::Dynamic, Eigen::Dynamic > distance_matrix(
        this->dataset_ptr->size(), this->dataset_ptr->size()
    );

    for(int i = 0; i < this->dataset_ptr->size(); ++i) 
        distance_matrix.col(i) = (center_points.rowwise() - center_points.row(i)).rowwise().norm();

    distance_matrix = rbf_ptr->eval_matrix(distance_matrix);

    // resizing the coefficients vector:
    // #rows = #(center points); #columns = output dimension
    coefficients.resize(this->dataset_ptr->size(), this->dataset_ptr->labels_at(0).rows());

    for(int i = 0; i < coefficients.cols(); ++i) 
        coefficients.col(i) = distance_matrix.fullPivLu().solve(output_at_center.col(i));
    
}

template <typename Datum_t, typename Label_t>
void RBFN<Datum_t, Label_t>::fit(const std::vector<int> &fit_indices) {
    
}
  
} // namespace tudat_learn

#endif // TUDAT_LEARN_RBFN_TPP