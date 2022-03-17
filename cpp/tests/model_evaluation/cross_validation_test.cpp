/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#include <iostream>
#include <type_traits>
#include <cmath>

#include <Eigen/Core>

#include <tudat-learn/dataset.hpp>
#include <tudat-learn/estimators/regressor.hpp>
#include <tudat-learn/estimators/regressors/rbfn.hpp>
#include <tudat-learn/model_evaluation/cross_validation.hpp>
#include <tudat-learn/model_evaluation/splits/k_fold_split.hpp>
#include <tudat-learn/types.hpp>


// template <typename Datum_t, typename Label_t, 
//   typename std::enable_if_t<tudat_learn::is_floating_point_eigen_vector<Label_t>::value>
// > double mean_absolute_percentage_error(
//   const std::shared_ptr<tudat_learn::Dataset<Datum_t, Label_t> > &dataset_ptr,
//   const std::shared_ptr<tudat_learn::Estimator<Datum_t, Label_t> > &estimator_ptr, 
//   const std::vector<size_t> &eval_indices
// ) {

//   Label_t vector_mape = Label_t::Zero(dataset_ptr.data_at(eval_indices.at(0)).rows());

//   for(size_t i = 0u; i < eval_indices.size(); ++i) 
//     vector_mape += ((dataset_ptr->labels_at(eval_indices.at(i)).array() - estimator_ptr->eval(dataset_ptr->data_at(eval_indices.at(i)).array)) / dataset_ptr->labels_at(eval_indices.at(i)).array()).array().abs();
    

//   vector_mape /= eval_indices.size();

//   // average of the error across all labels is returned
//   double mape = vector_mape.mean();

//   return mape;
// }

double mean_absolute_percentage_error(
  const std::shared_ptr<tudat_learn::Dataset<Eigen::VectorXd, Eigen::VectorXd> > &dataset_ptr,
  const std::shared_ptr<tudat_learn::Estimator<Eigen::VectorXd, Eigen::VectorXd> > &estimator_ptr, 
  const std::vector<size_t> &eval_indices
) {

  Eigen::VectorXd vector_mape = Eigen::VectorXd::Zero(dataset_ptr->data_at(eval_indices.at(0)).rows());

  for(size_t i = 0u; i < eval_indices.size(); ++i) {
    vector_mape += (
      (
        dataset_ptr->labels_at(eval_indices.at(i)).array() - 
        estimator_ptr->eval(dataset_ptr->data_at(eval_indices.at(i))).array()
      ) / dataset_ptr->labels_at(eval_indices.at(i)).array()
    ).array().abs().matrix();
  }
    

  vector_mape /= eval_indices.size();

  // average of the error across all labels is returned
  double mape = vector_mape.mean();

  return mape;
}

int main( ) {

  std::vector< Eigen::VectorXd > center_points({
    (Eigen::VectorXd(7) << 0.548814, 0.715189, 0.602763, 0.544883, 0.423655, 0.645894, 0.437587).finished(),
    (Eigen::VectorXd(7) << 0.891773, 0.963663, 0.383442, 0.791725, 0.528895, 0.568045, 0.925597).finished(),
    (Eigen::VectorXd(7) << 0.071036, 0.087129, 0.020218, 0.832620, 0.778157, 0.870012, 0.978618).finished(),
    (Eigen::VectorXd(7) << 0.799159, 0.461479, 0.780529, 0.118274, 0.639921, 0.143353, 0.944669).finished(),
    (Eigen::VectorXd(7) << 0.521848, 0.414662, 0.264556, 0.774234, 0.456150, 0.568434, 0.018790).finished(),
    (Eigen::VectorXd(7) << 0.617635, 0.612096, 0.616934, 0.943748, 0.681820, 0.359508, 0.437032).finished(),
    (Eigen::VectorXd(7) << 0.697631, 0.060225, 0.666767, 0.670638, 0.210383, 0.128926, 0.315428).finished(),
    (Eigen::VectorXd(7) << 0.363711, 0.570197, 0.438602, 0.988374, 0.102045, 0.208877, 0.161310).finished(),
    (Eigen::VectorXd(7) << 0.653108, 0.253292, 0.466311, 0.244426, 0.158970, 0.110375, 0.656330).finished(),
    (Eigen::VectorXd(7) << 0.138183, 0.196582, 0.368725, 0.820993, 0.097101, 0.837945, 0.096098).finished()
  });
  
  std::vector< Eigen::VectorXd > labels({
    (Eigen::VectorXd(2) << 0.976459, 0.468651).finished(),
    (Eigen::VectorXd(2) << 0.976761, 0.604846).finished(),
    (Eigen::VectorXd(2) << 0.739264, 0.039188).finished(),
    (Eigen::VectorXd(2) << 0.282807, 0.120197).finished(),
    (Eigen::VectorXd(2) << 0.296140, 0.118728).finished(),
    (Eigen::VectorXd(2) << 0.317983, 0.414263).finished(),
    (Eigen::VectorXd(2) << 0.064147, 0.692472).finished(),
    (Eigen::VectorXd(2) << 0.566601, 0.265389).finished(),
    (Eigen::VectorXd(2) << 0.523248, 0.093941).finished(),
    (Eigen::VectorXd(2) << 0.575946, 0.929296).finished()
  });

  double sigma = 0.318569;

  auto dataset_ptr = std::make_shared< tudat_learn::Dataset<Eigen::VectorXd, Eigen::VectorXd> >(tudat_learn::Dataset(center_points, labels));
  auto gaussian_rbf_ptr = std::make_shared< tudat_learn::GaussianRBF<double> >(tudat_learn::GaussianRBF<double>(sigma));

  tudat_learn::RBFN<Eigen::VectorXd, Eigen::VectorXd> rbfn(dataset_ptr, gaussian_rbf_ptr);

  std::vector< std::function<double(
    const std::shared_ptr<tudat_learn::Dataset<Eigen::VectorXd, Eigen::VectorXd>> &,
    const std::shared_ptr<tudat_learn::Estimator<Eigen::VectorXd, Eigen::VectorXd>> &, 
    const std::vector<size_t> &
  )> > metrics;

  metrics.push_back(&mean_absolute_percentage_error);

  tudat_learn::CrossValidation<Eigen::VectorXd, Eigen::VectorXd> cv(
    dataset_ptr,
    std::make_shared< tudat_learn::RBFN<Eigen::VectorXd, Eigen::VectorXd> >(rbfn),
    std::make_shared< tudat_learn::KFoldSplit<Eigen::VectorXd, Eigen::VectorXd> >(dataset_ptr, 3, false),
    metrics
  );

  // cv.cross_validate( );

  return 0;
}