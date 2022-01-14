/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#include <iomanip>
#include <iostream>
#include <memory>
#include <vector>

#include <Eigen/Core>

#include "tudat-learn/dataset.hpp"
#include "tudat-learn/estimators/regressors/rbf.hpp"
#include "tudat-learn/estimators/regressors/rbfn.hpp"

namespace tudat_learn{ 

// template <typename Datum_t, typename Label_t>
// class DerivativeTester : public tudat_learn::RBFN<Datum_t, Label_t> {

//   public:
//     DerivativeTester(
//       const std::shared_ptr< Dataset<Datum_t, Label_t> > &dataset_ptr,
//       const std::shared_ptr< RBF<typename Label_t::Scalar> > &rbf_ptr
//     ) : RBFN<Datum_t, Label_t>(dataset_ptr, rbf_ptr) { 
//       this->coefficients = Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic>::Ones(
//         dataset_ptr->size(), dataset_ptr->labels_at(0).rows()
//       );

//      this->center_points.resize(this->dataset_ptr->size(), this->dataset_ptr->data_at(0).rows());

//       for(int i = 0; i < this->dataset_ptr->size(); ++i) {
//         this->center_points.row(i)    = this->dataset_ptr->data_at(i);
//       }

//     }

//     // virtual void fit( ) override {
//     //   RBFN<Datum_t, Label_t>::fit();

//     //   this->coefficients = Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic>::Ones(
//     //     this->coefficients.rows(), this->coefficients.cols()
//     //     );
//     // }
// };

template <typename Datum_t, typename Label_t>
class DerivativeTester : public tudat_learn::RBFN<Datum_t, Label_t> {

  public:
    DerivativeTester(
      const std::shared_ptr< Dataset<Datum_t, Label_t> > &dataset_ptr,
      const std::shared_ptr< RBF<typename Label_t::Scalar> > &rbf_ptr
    ) : RBFN<Datum_t, Label_t>(dataset_ptr, rbf_ptr) { 
      this->coefficients = Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic>::Ones(
        dataset_ptr->size(), dataset_ptr->labels_at(0).rows()
      );

     this->center_points.resize(this->dataset_ptr->size(), this->dataset_ptr->data_at(0).rows());

      for(int i = 0; i < this->dataset_ptr->size(); ++i) {
        this->center_points.row(i)    = this->dataset_ptr->data_at(i);
      }
    }

    virtual Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> gradient(const Datum_t &x) const override {
      using MatrixX = Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic>;

      MatrixX gradient = MatrixX::Zero(
        this->coefficients.cols(), this->center_points.cols()
      );

      for(int k = 0; k < this->coefficients.cols(); ++k) {
          MatrixX pd = MatrixX::Zero(
            this->center_points.rows(), this->center_points.cols()
          );
          
          for(int n = 0; n < this->center_points.rows(); ++n) {
            pd.row(n) = *(this->rbf_ptr->eval_gradient(x, this->center_points.row(n).transpose()));
          }

          gradient.row(k) = this->coefficients.col(k).transpose() * pd;

      }
      
      return gradient;
    }
};

template <typename Datum_t, typename Label_t>
class DerivativeTesterPolynomial : public tudat_learn::RBFNPolynomial<Datum_t, Label_t> {

  public:
    DerivativeTesterPolynomial(
      const std::shared_ptr< Dataset<Datum_t, Label_t> > &dataset_ptr,
      const std::shared_ptr< RBF<typename Label_t::Scalar> > &rbf_ptr
    ) : RBFNPolynomial<Datum_t, Label_t>(dataset_ptr, rbf_ptr) { }

    virtual Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> gradient(const Datum_t &x) const override {
      using MatrixX = Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic>;

      MatrixX gradient = MatrixX::Zero(
        this->coefficients.cols(), this->center_points.cols()
      );

      for(int k = 0; k < this->coefficients.cols(); ++k) {
          MatrixX pd = MatrixX::Zero(
            this->center_points.rows(), this->center_points.cols()
          );
          
          for(int n = 0; n < this->center_points.rows(); ++n) {
            pd.row(n) = *(this->rbf_ptr->eval_gradient(x, this->center_points.row(n).transpose()));
          }

          gradient.row(k) = this->coefficients.col(k).head(this->center_points.rows()).transpose() * pd;

      }

      gradient += this->coefficients.block(this->center_points.rows() + 1, 0, this->center_points.cols(), this->coefficients.cols()).transpose();
      
      return gradient;
    }
};



} // namespace tudat_learn

int main( ) {
  
  // Values generated using /tudat-learn/cpp/tests/python_scripts/estimators/regressors/rbfn_test.py
  std::cout << std::setprecision(6) << std::fixed;

  std::vector< Eigen::VectorXf > data({
    (Eigen::VectorXf(7) << 0.548814, 0.715189, 0.602763, 0.544883, 0.423655, 0.645894, 0.437587).finished(),
    (Eigen::VectorXf(7) << 0.891773, 0.963663, 0.383442, 0.791725, 0.528895, 0.568045, 0.925597).finished(),
    (Eigen::VectorXf(7) << 0.071036, 0.087129, 0.020218, 0.832620, 0.778157, 0.870012, 0.978618).finished(),
    (Eigen::VectorXf(7) << 0.799159, 0.461479, 0.780529, 0.118274, 0.639921, 0.143353, 0.944669).finished(),
    (Eigen::VectorXf(7) << 0.521848, 0.414662, 0.264556, 0.774234, 0.456150, 0.568434, 0.018790).finished(),
    (Eigen::VectorXf(7) << 0.617635, 0.612096, 0.616934, 0.943748, 0.681820, 0.359508, 0.437032).finished(),
    (Eigen::VectorXf(7) << 0.697631, 0.060225, 0.666767, 0.670638, 0.210383, 0.128926, 0.315428).finished(),
    (Eigen::VectorXf(7) << 0.363711, 0.570197, 0.438602, 0.988374, 0.102045, 0.208877, 0.161310).finished(),
    (Eigen::VectorXf(7) << 0.653108, 0.253292, 0.466311, 0.244426, 0.158970, 0.110375, 0.656330).finished(),
    (Eigen::VectorXf(7) << 0.138183, 0.196582, 0.368725, 0.820993, 0.097101, 0.837945, 0.096098).finished()
  });
  
  std::vector< Eigen::VectorXf > labels({
    (Eigen::VectorXf(2) << 0.976459, 0.468651).finished(),
    (Eigen::VectorXf(2) << 0.976761, 0.604846).finished(),
    (Eigen::VectorXf(2) << 0.739264, 0.039188).finished(),
    (Eigen::VectorXf(2) << 0.282807, 0.120197).finished(),
    (Eigen::VectorXf(2) << 0.296140, 0.118728).finished(),
    (Eigen::VectorXf(2) << 0.317983, 0.414263).finished(),
    (Eigen::VectorXf(2) << 0.064147, 0.692472).finished(),
    (Eigen::VectorXf(2) << 0.566601, 0.265389).finished(),
    (Eigen::VectorXf(2) << 0.523248, 0.093941).finished(),
    (Eigen::VectorXf(2) << 0.575946, 0.929296).finished()
  });


  float sigma = 0.318569;

  auto dataset_ptr = std::make_shared< tudat_learn::Dataset<Eigen::VectorXf, Eigen::VectorXf> >(tudat_learn::Dataset(data, labels));
  auto cubic_rbf_ptr = std::make_shared< tudat_learn::CubicRBF<float> >(tudat_learn::CubicRBF<float>());
  auto gaussian_rbf_ptr = std::make_shared< tudat_learn::GaussianRBF<float> >(tudat_learn::GaussianRBF<float>(sigma));

  tudat_learn::RBFN<Eigen::VectorXf, Eigen::VectorXf> cubic_rbfn(dataset_ptr, cubic_rbf_ptr);
  cubic_rbfn.fit();
  std::cout << "Coefficients of the Cubic RBFN:\n" << cubic_rbfn.get_coefficients() << std::endl;

  Eigen::MatrixXf cubic_coefficients_expected(10,2);
  cubic_coefficients_expected <<  -1.337114, -2.755163,
                                  -0.212870, 2.093877,
                                  0.527279, -0.079324,
                                  -0.428405, -1.163986,
                                  1.476176, 4.215983,
                                  0.801391, -1.680252,
                                  0.322220, 0.782600,
                                  -0.267450, -1.772963,
                                  0.935899, 1.639483,
                                  -1.084187, -1.09602;

  if( !cubic_coefficients_expected.isApprox(cubic_rbfn.get_coefficients()) )
    return 1;

  tudat_learn::RBFN<Eigen::VectorXf, Eigen::VectorXf> gaussian_rbfn(dataset_ptr, gaussian_rbf_ptr);
  gaussian_rbfn.fit();
  std::cout << "Coefficients of the Gaussian RBFN:\n" << gaussian_rbfn.get_coefficients() << std::endl;

  Eigen::MatrixXf gaussian_coefficients_expected(10,2);
  gaussian_coefficients_expected <<  0.955820, 0.447495,
                                     0.971372, 0.601469,
                                     0.739263, 0.039187,
                                     0.278673, 0.119504,
                                     0.260680, 0.087732,
                                     0.269842, 0.390772,
                                     0.048714, 0.688877,
                                     0.557143, 0.256347,
                                     0.519257, 0.076439,
                                     0.570117, 0.927115;

  if( !gaussian_coefficients_expected.isApprox(gaussian_rbfn.get_coefficients()) )
    return 1;

  std::vector< Eigen::VectorXf > inputs({
    (Eigen::VectorXf(7) << 0.667410, 0.131798, 0.716327, 0.289406, 0.183191, 0.586513, 0.020108).finished(),
    (Eigen::VectorXf(7) << 0.828940, 0.004695, 0.677817, 0.270008, 0.735194, 0.962189, 0.248753).finished(),
    (Eigen::VectorXf(7) << 0.576157, 0.592042, 0.572252, 0.223082, 0.952749, 0.447125, 0.846409).finished()
  });

  Eigen::MatrixXf cubic_output(3,2);
  cubic_output = cubic_rbfn.eval(inputs);
  std::cout << "Cubic RBFN output:\n" << cubic_output << std::endl;

  Eigen::MatrixXf cubic_output_expected(3, 2);
  cubic_output_expected << 1.349381,  0.979610,
                           0.858228, -0.700269,
                           0.105554,  0.109006;

  if( !cubic_output_expected.isApprox(cubic_output) )
    return 1;

  for(int i = 0; i < inputs.size(); ++i)
    if( !cubic_rbfn.eval(inputs[i]).isApprox(cubic_output_expected.row(i).transpose()) )
      return 1;

  Eigen::MatrixXf gaussian_output(3,2);
  gaussian_output = gaussian_rbfn.eval(inputs);
  std::cout << "Gaussian RBFN output:\n" << gaussian_output << std::endl;

  Eigen::MatrixXf gaussian_output_expected(3, 2);
  gaussian_output_expected << 3.301947e-03, 9.486395e-03,
                              1.595987e-04, 8.165682e-05,
                              1.47863e-02,  6.625883e-03;

  if( !gaussian_output_expected.isApprox(gaussian_output) )
    return 1;

  for(int i = 0; i < inputs.size(); ++i)
    if( !gaussian_rbfn.eval(inputs[i]).isApprox(gaussian_output_expected.row(i).transpose()) )
      return 1;

  // Testing fit(const std::vector<int> &fit_indices)
  std::vector< Eigen::VectorXf > data_extra({
    (Eigen::VectorXf(7) << 1, 1, 2, 0, 0, 0, 0).finished(),
    (Eigen::VectorXf(7) << 0.548814, 0.715189, 0.602763, 0.544883, 0.423655, 0.645894, 0.437587).finished(),
    (Eigen::VectorXf(7) << 0.891773, 0.963663, 0.383442, 0.791725, 0.528895, 0.568045, 0.925597).finished(),
    (Eigen::VectorXf(7) << 0.071036, 0.087129, 0.020218, 0.832620, 0.778157, 0.870012, 0.978618).finished(),
    (Eigen::VectorXf(7) << 0.799159, 0.461479, 0.780529, 0.118274, 0.639921, 0.143353, 0.944669).finished(),
    (Eigen::VectorXf(7) << 0.521848, 0.414662, 0.264556, 0.774234, 0.456150, 0.568434, 0.018790).finished(),
    (Eigen::VectorXf(7) << 0.617635, 0.612096, 0.616934, 0.943748, 0.681820, 0.359508, 0.437032).finished(),
    (Eigen::VectorXf(7) << 0.697631, 0.060225, 0.666767, 0.670638, 0.210383, 0.128926, 0.315428).finished(),
    (Eigen::VectorXf(7) << 0.363711, 0.570197, 0.438602, 0.988374, 0.102045, 0.208877, 0.161310).finished(),
    (Eigen::VectorXf(7) << 0.653108, 0.253292, 0.466311, 0.244426, 0.158970, 0.110375, 0.656330).finished(),
    (Eigen::VectorXf(7) << 0.138183, 0.196582, 0.368725, 0.820993, 0.097101, 0.837945, 0.096098).finished(),
    (Eigen::VectorXf(7) << 0, 0, 0, 0, 0, 0, 0).finished()
  });

  std::vector< Eigen::VectorXf > labels_extra({
    (Eigen::VectorXf(2) << 1, 0).finished(),
    (Eigen::VectorXf(2) << 0.976459, 0.468651).finished(),
    (Eigen::VectorXf(2) << 0.976761, 0.604846).finished(),
    (Eigen::VectorXf(2) << 0.739264, 0.039188).finished(),
    (Eigen::VectorXf(2) << 0.282807, 0.120197).finished(),
    (Eigen::VectorXf(2) << 0.296140, 0.118728).finished(),
    (Eigen::VectorXf(2) << 0.317983, 0.414263).finished(),
    (Eigen::VectorXf(2) << 0.064147, 0.692472).finished(),
    (Eigen::VectorXf(2) << 0.566601, 0.265389).finished(),
    (Eigen::VectorXf(2) << 0.523248, 0.093941).finished(),
    (Eigen::VectorXf(2) << 0.575946, 0.929296).finished(),
    (Eigen::VectorXf(2) << 0, 0).finished()
  });
  
  auto dataset_extra_ptr = std::make_shared< tudat_learn::Dataset<Eigen::VectorXf, Eigen::VectorXf> >(tudat_learn::Dataset(data_extra, labels_extra));

  tudat_learn::RBFN<Eigen::VectorXf, Eigen::VectorXf> cubic_rbfn_extra(dataset_extra_ptr, cubic_rbf_ptr);
  cubic_rbfn_extra.fit(std::vector<int>({1,2,3,4,5,6,7,8,9,10}));
  std::cout << "Coefficients of the Cubic RBFN when fitting to specific indices:\n" 
            << cubic_rbfn_extra.get_coefficients() << std::endl;

  if( !cubic_coefficients_expected.isApprox(cubic_rbfn_extra.get_coefficients()) )
    return 1;

  tudat_learn::RBFN<Eigen::VectorXf, Eigen::VectorXf> gaussian_rbfn_extra(dataset_extra_ptr, gaussian_rbf_ptr);
  gaussian_rbfn_extra.fit(std::vector<int>({1,2,3,4,5,6,7,8,9,10}));
  std::cout << "Coefficients of the Gaussian RBFN when fitting to specific indices:\n" 
            << gaussian_rbfn_extra.get_coefficients() << std::endl;

  if( !gaussian_coefficients_expected.isApprox(gaussian_rbfn_extra.get_coefficients()) )
    return 1;

  // Testing RBFNPolynomial
  tudat_learn::RBFNPolynomial<Eigen::VectorXf, Eigen::VectorXf> cubic_rbfn_poly(dataset_ptr, cubic_rbf_ptr);
  cubic_rbfn_poly.fit();
  std::cout << "Coefficients of the Cubic RBFNPolynomial:\n" << cubic_rbfn_poly.get_coefficients() << std::endl;

  Eigen::MatrixXf cubic_coefficients_expected_poly(18,2);
  cubic_coefficients_expected_poly <<  0.210345, -0.124317,
                                      -0.047379,  0.017144,
                                       0.061512, -0.046473,
                                      -0.112250,  0.019956,
                                      -0.019254, -0.009396,
                                      -0.058411,  0.108840,
                                       0.100389, -0.089431,
                                       0.014692, -0.048700,
                                       0.006936,  0.066522,
                                      -0.156580,  0.105854,
                                       0.543433, -1.464384,
                                      -0.472862,  0.854898,
                                       0.850252, -0.588660,
                                      -0.088981,  1.370274,
                                      -0.243245,  0.940585,
                                      -0.722858, -1.034396,
                                       0.615800,  1.275494,
                                       0.524939,  0.396612;

  if( !cubic_coefficients_expected_poly.isApprox(cubic_rbfn_poly.get_coefficients()) )
    return 1;

  tudat_learn::RBFNPolynomial<Eigen::VectorXf, Eigen::VectorXf> gaussian_rbfn_poly(dataset_ptr, gaussian_rbf_ptr);
  gaussian_rbfn_poly.fit();
  std::cout << "Coefficients of the Gaussian RBFNPolynomial:\n" << gaussian_rbfn_poly.get_coefficients() << std::endl;

  Eigen::MatrixXf gaussian_coefficients_expected_poly(18,2);
  gaussian_coefficients_expected_poly <<  0.164707, -0.080970,
                                         -0.043197,  0.016828,
                                          0.042483, -0.024992,
                                         -0.113948,  0.037186,
                                         -0.026745,  0.004714,
                                         -0.003996,  0.032135,
                                          0.061702, -0.042552,
                                         -0.010971, -0.010852,
                                          0.045096,  0.006501,
                                         -0.115131,  0.062003,
                                          0.360567, -1.450516,
                                         -0.397730,  0.858258,
                                          0.820341, -0.567872,
                                         -0.171501,  1.422127,
                                         -0.153605,  0.888652,
                                         -0.674028, -1.052130,
                                          0.513273,  1.337520,
                                          0.513274,  0.398121;

  if( !gaussian_coefficients_expected_poly.isApprox(gaussian_rbfn_poly.get_coefficients()) )
    return 1;

  Eigen::MatrixXf cubic_output_poly(3,2);
  cubic_output_poly = cubic_rbfn_poly.eval(inputs);
  std::cout << "Cubic RBFNPolynomial output:\n" << cubic_output_poly << std::endl;

  Eigen::MatrixXf cubic_output_expected_poly(3, 2);
  cubic_output_expected_poly << 0.337597,  0.844558,
                                0.071872,  1.007463,
                                0.486969, -0.348277;

  if( !cubic_output_expected_poly.isApprox(cubic_output_poly) )
    return 1;

  for(int i = 0; i < inputs.size(); ++i) {
    if( !cubic_rbfn_poly.eval(inputs[i]).isApprox(cubic_output_expected_poly.row(i).transpose()) )
      return 1;
  }

  Eigen::MatrixXf gaussian_output_poly(3,2);
  gaussian_output_poly = gaussian_rbfn_poly.eval(inputs);
  std::cout << "gaussian RBFNPolynomial output:\n" << gaussian_output_poly << std::endl;

  Eigen::MatrixXf gaussian_output_expected_poly(3, 2);
  gaussian_output_expected_poly << 0.224686,  0.922516,
                                   0.003029,  1.074597,
                                   0.502017, -0.346183;

  if( !gaussian_output_expected_poly.isApprox(gaussian_output_poly) )
    return 1;

  for(int i = 0; i < inputs.size(); ++i) {
    if( !gaussian_rbfn_poly.eval(inputs[i]).isApprox(gaussian_output_expected_poly.row(i).transpose()) )
      return 1;
  }

  tudat_learn::RBFNPolynomial<Eigen::VectorXf, Eigen::VectorXf> cubic_rbfn_extra_poly(dataset_extra_ptr, cubic_rbf_ptr);
  cubic_rbfn_extra_poly.fit(std::vector<int>({1,2,3,4,5,6,7,8,9,10}));
  std::cout << "Coefficients of the Cubic RBFNPolynomial when fitting to specific indices:\n" 
            << cubic_rbfn_extra_poly.get_coefficients() << std::endl;

  if( !cubic_coefficients_expected_poly.isApprox(cubic_rbfn_extra_poly.get_coefficients()) )
    return 1;

  tudat_learn::RBFNPolynomial<Eigen::VectorXf, Eigen::VectorXf> gaussian_rbfn_extra_poly(dataset_extra_ptr, gaussian_rbf_ptr);
  gaussian_rbfn_extra_poly.fit(std::vector<int>({1,2,3,4,5,6,7,8,9,10}));
  std::cout << "Coefficients of the Gaussian RBFNPolynomial when fitting to specific indices:\n" 
            << gaussian_rbfn_extra_poly.get_coefficients() << std::endl;

  if( !gaussian_coefficients_expected_poly.isApprox(gaussian_rbfn_extra_poly.get_coefficients()) )
    return 1;

  // Testing Derivatives
  // std::vector< Eigen::VectorXf > data_derivatives({
  //   (Eigen::VectorXf(3) << 0.25891675, 0.51127472, 0.40493414).finished()
  // });
  
  // std::vector< Eigen::VectorXf > labels_derivatives({
  //   (Eigen::VectorXf(2) << 0.25670216, 0.25670216).finished()
  // });

  // Eigen::VectorXf x_derivatives = (Eigen::VectorXf(3) << 0.84442185, 0.7579544,  0.42057158).finished();

  // auto dataset_ptr_derivatives = std::make_shared< tudat_learn::Dataset<Eigen::VectorXf, Eigen::VectorXf> >(tudat_learn::Dataset(data_derivatives, labels_derivatives));
  // auto gaussian_rbf_ptr_derivatives = std::make_shared< tudat_learn::GaussianRBF<float> >(tudat_learn::GaussianRBF<float>(0.78379858));
  
  // tudat_learn::DerivativeTester<Eigen::VectorXf, Eigen::VectorXf> derivative_tester_cubic(dataset_ptr_derivatives, cubic_rbf_ptr);
  // tudat_learn::DerivativeTester<Eigen::VectorXf, Eigen::VectorXf> derivative_tester_gaussian(dataset_ptr_derivatives, gaussian_rbf_ptr_derivatives);

  // std::cout << "Gradient of the Cubic RBF:\n" << derivative_tester_cubic.gradient(x_derivatives) << std::endl;
  // std::cout << "Gradient of the Gaussian RBF:\n" << derivative_tester_gaussian.gradient(x_derivatives) << std::endl;

  // Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> dif(1, 3);
  // Eigen::Matrix<float, Eigen::Dynamic, 1> dist(1);
  // dif = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>::Random(1,3);
  // dist = Eigen::Matrix<float, Eigen::Dynamic, 1>::Random(1);

  // std::cout << "Hessian of the Cubic RBF:\n" << derivative_tester_cubic.hessians(x_derivatives)[0] << std::endl;
  // std::cout << "Hessian of the Gaussian RBF 0:\n" << derivative_tester_gaussian.hessians(x_derivatives)[0] << std::endl;
  // std::cout << "Hessian of the Gaussian RBF 1:\n" << derivative_tester_gaussian.hessians(x_derivatives)[1] << std::endl;

  tudat_learn::DerivativeTester<Eigen::VectorXf, Eigen::VectorXf> derivative_tester_cubic(dataset_ptr, cubic_rbf_ptr);
  tudat_learn::DerivativeTester<Eigen::VectorXf, Eigen::VectorXf> derivative_tester_gaussian(dataset_ptr, gaussian_rbf_ptr);

  derivative_tester_cubic.fit();
  derivative_tester_gaussian.fit();

  std::cout << "Gradient of the Cubic Derivative Tester at input[0]:\n" << derivative_tester_cubic.gradient(inputs.at(0)) << std::endl;
  std::cout << "Gradient of the Cubic RBFN at input[0]:\n" << cubic_rbfn.gradient(inputs.at(0)) << std::endl;
  for(int i = 0; i < inputs.size(); ++i) {
    if(!derivative_tester_cubic.gradient(inputs[i]).isApprox(cubic_rbfn.gradient(inputs.at(i))))
      return 1;
  }

  std::cout << "Gradient of the Gaussian Derivative Tester at input[0]:\n" << derivative_tester_gaussian.gradient(inputs.at(0)) << std::endl;
  std::cout << "Gradient of the Gaussian RBFN at input[0]:\n" << gaussian_rbfn.gradient(inputs.at(0)) << std::endl;

  tudat_learn::DerivativeTesterPolynomial<Eigen::VectorXf, Eigen::VectorXf> derivative_tester_cubic_poly(dataset_ptr, cubic_rbf_ptr);
  tudat_learn::DerivativeTesterPolynomial<Eigen::VectorXf, Eigen::VectorXf> derivative_tester_gaussian_poly(dataset_ptr, gaussian_rbf_ptr);

  derivative_tester_cubic_poly.fit();
  derivative_tester_gaussian_poly.fit();

  std::cout << "Gradient of the Polynomial Cubic Derivative Tester at input[0]:\n" << derivative_tester_cubic_poly.gradient(inputs.at(0)) << std::endl;
  std::cout << "Gradient of the Polynomial Cubic RBFN at input[0]:\n" << cubic_rbfn_poly.gradient(inputs.at(0)) << std::endl;  
  std::cout << "Gradient of the Polynomial Gaussian Derivative Tester at input[0]:\n" << derivative_tester_gaussian_poly.gradient(inputs.at(0)) << std::endl;
  std::cout << "Gradient of the Polynomial Gaussian RBFN at input[0]:\n" << gaussian_rbfn_poly.gradient(inputs.at(0)) << std::endl;  
  

  return 0;
}