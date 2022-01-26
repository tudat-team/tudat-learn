#include <iostream>
#include <memory>
#include <typeinfo>
#include <vector>

#include <Eigen/Core>

#include "tudat-learn/dataset.hpp"
#include "tudat-learn/estimators/regressors/rbf.hpp"
#include "tudat-learn/estimators/regressors/rbfn.hpp"
#include "tudat-learn/types.hpp"


  
// Copyright 2018 Delft University of Technology
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <chrono>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <type_traits>

/// @brief A timer using the C++11 high resolution monotonic clock.
struct Timer {
  using system_clock = std::chrono::system_clock;
  using nanoseconds = std::chrono::nanoseconds;
  using time_point = std::chrono::time_point<system_clock, nanoseconds>;
  using duration = std::chrono::duration<double>;

  Timer() = default;

  /// @brief Timer start point.
  time_point start_{};
  /// @brief Timer stop point.
  time_point stop_{};

  /// @brief Start the timer.
  inline void start() { start_ = std::chrono::high_resolution_clock::now(); }

  /// @brief Stop the timer.
  inline void stop() { stop_ = std::chrono::high_resolution_clock::now(); }

  /// @brief Retrieve the interval in seconds.
  double seconds() {
    duration diff = stop_ - start_;
    return diff.count();
  }

  /// @brief Return the interval in seconds as a formatted string.
  std::string str(int width = 14) {
    std::stringstream ss;
    ss << std::setprecision(width - 5) << std::setw(width) << std::fixed << seconds();
    return ss.str();
  }

  /// @brief Print the interval on some output stream
  void report(std::ostream &os = std::cout, bool last = false, int width = 15) {
    os << std::setw(width) << ((last ? " " : "") + str() + (last ? "\n" : ",")) << std::flush;
  }
};

namespace tudat_learn {

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

    virtual std::vector< Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> > hessians(const Datum_t &x) const override {
      using MatrixX = Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic>;

      std::vector<MatrixX> hessians(
        this->coefficients.cols(),
        MatrixX::Zero(this->center_points.cols(), this->center_points.cols())
      );

      for(int n = 0; n < this->center_points.rows(); ++n) {
        MatrixX hessian_rbf(
          this->center_points.cols(), this->center_points.cols()
        );

        hessian_rbf = *(this->rbf_ptr->eval_hessian(x, this->center_points.row(n)));

        for(int k = 0; k < this->coefficients.cols(); k++) {
          hessians.at(k) += hessian_rbf * this->coefficients(n, k);
        }
      }

      return hessians;
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

    virtual std::vector< Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic> > hessians(const Datum_t &x) const override {
      using MatrixX = Eigen::Matrix<typename Datum_t::Scalar, Eigen::Dynamic, Eigen::Dynamic>;

      std::vector<MatrixX> hessians(
        this->coefficients.cols(),
        MatrixX::Zero(this->center_points.cols(), this->center_points.cols())
      );

      for(int n = 0; n < this->center_points.rows(); ++n) {
        MatrixX hessian_rbf(
          this->center_points.cols(), this->center_points.cols()
        );

        hessian_rbf = *(this->rbf_ptr->eval_hessian(x, this->center_points.row(n)));

        for(int k = 0; k < this->coefficients.cols(); k++) {
          hessians.at(k) += hessian_rbf * this->coefficients(n, k);
        }
      }

      return hessians;
    } 
};

} // namespace tudat_learn



int main()
{
  int integer = 1;
  float f = 2.345;
  float f2 = integer * f;
  std::cout << f2 << std::endl;
  std::cout << integer * f << std::endl;

  Eigen::VectorXf v1 = Eigen::VectorXf::Random(6);
  Eigen::VectorXf v2 = Eigen::VectorXf::Random(6);
  std::cout << "v1:\n" << v1.transpose() << std::endl;
  std::cout << "v2:\n" << v2.transpose() << std::endl;
  std::cout << "Elementwise max:\n" << v1.array().max(v2.array()).transpose() << std::endl;
  std::cout << "Elementwise max:\n" << v2.array().max(v1.array()).transpose() << std::endl;
  std::cout << "Elementwise min:\n" << v2.array().min(v1.array()).transpose() << std::endl;

  using VectorX = Eigen::Matrix<float, Eigen::Dynamic, 1>;

  int dataset_size = 100;
  int input_size = 100;
  int output_size = 1;
  int repetitions = 100;

  std::vector<VectorX> data(dataset_size);
  for(auto &it: data)
    it = VectorX::Random(input_size);

  std::vector<VectorX> labels(dataset_size);
  for(auto &it: labels)
    it = VectorX::Random(output_size);

  auto dataset_ptr = std::make_shared< tudat_learn::Dataset<VectorX, VectorX> >(tudat_learn::Dataset(data, labels));
  auto rbf_ptr = std::make_shared< tudat_learn::CubicRBF<float> >(tudat_learn::CubicRBF<float>());
  // auto rbf_ptr = std::make_shared< tudat_learn::GaussianRBF<float> >(tudat_learn::GaussianRBF<float>(sigma));

  tudat_learn::RBFN<VectorX, VectorX> rbfn(dataset_ptr, rbf_ptr);  
  tudat_learn::DerivativeTester<VectorX, VectorX> tester(dataset_ptr, rbf_ptr);  
  Timer t;

  t.start();
  rbfn.fit();
  t.stop();
  std::cout << "RBFN Fit takes " << t.seconds() << " seconds." << std::endl;

  tester.fit();

  
  t.start();
  for(int i = 0; i < repetitions; ++i) {
    rbfn.gradient(VectorX::Random(input_size));
  }
  t.stop();
  std::cout << "RBFN Gradient takes " << t.seconds() << " seconds." << std::endl;

  t.start();
  for(int i = 0; i < repetitions; ++i) {
    tester.gradient(VectorX::Random(input_size));
  }
  t.stop();
  std::cout << "Tester Gradient takes " << t.seconds() << " seconds." << std::endl;

  t.start();
  for(int i = 0; i < repetitions; ++i) {
    rbfn.hessians(VectorX::Random(input_size));
  }
  t.stop();
  std::cout << "RBFN Hessians takes " << t.seconds() << " seconds." << std::endl;

  t.start();
  for(int i = 0; i < repetitions; ++i) {
    tester.hessians(VectorX::Random(input_size));
  }
  t.stop();
  std::cout << "Tester Hessians takes " << t.seconds() << " seconds." << std::endl;

  // Timer t;
  // t.start();
  // for(int i = 0; i < 100000; ++i) {
  //   ev3 = ev1 + ev2;
  // }
  // t.stop();
  // std::cout << "Adding Eigen::Vectors takes " << t.seconds() << " seconds." << std::endl;

  

  return 0;
}