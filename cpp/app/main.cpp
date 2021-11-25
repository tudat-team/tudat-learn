#include <iostream>
#include <memory>
#include <typeinfo>
#include <vector>

#include <Eigen/Core>

#include "tudat-learn/dataset.hpp"
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

// typedef class Null_t {

// } null_t;


// template< typename Datum_t, typename Target_t = null_t >
// class Dataset {
//   public:
//     Dataset(const std::vector< Datum_t > &data) : data(data) {
//       std::cout << "Only Data." << std::endl;
//     }

//     Dataset(const std::vector< Datum_t > &data, const std::vector< Target_t > &targets) 
//     : data(data), targets(targets) { 
//       std::cout << "Data and Targets." << std::endl;
//     }

//     std::vector< Datum_t > data;

//     std::vector< Target_t > targets;

// };



// class BaseDataset{
//   public:
//     // virtual void print_type() const {
//     //   std::cout << "Base Class" << std::endl;
//     // }
//     virtual void print_type() const = 0;

// };

// template <typename Datum_t>
// class UnlabelledDataset : public BaseDataset {
//   public:
//     UnlabelledDataset(const std::vector< Datum_t > &data) : data(data) {
//       std::cout << "Only Data." << std::endl;
//     }

//     void print_type() const {
//       std::cout << "Data is the type." << std::endl;
//     }

//     std::vector< Datum_t > data;
// };

// template <typename Datum_t, typename Target_t>
// class LabelledDataset : public BaseDataset {
//   public:
//     LabelledDataset(const std::vector< Datum_t > &data, const std::vector< Target_t > &targets) 
//       : data(data), targets(targets) { 
//         std::cout << "Data and Targets." << std::endl;
//     }

//     void print_type() const {
//       std::cout << "Data and Targets is the type." << std::endl;
//     }

//     void print_data() const {
//       for(const auto &it : data)
//         std::cout << it << ", ";
//       std::cout << std::endl;
//     }

//     std::vector< Datum_t > data;
//     std::vector< Target_t > targets;
// };

// class BaseOther {
//   public:
//     virtual void print_type(const BaseDataset &dataset) = 0;

//     // template< typename T, typename U >
//     // virtual void print_temp(const LabelledDataset<T, U> &dataset) = 0;
// };

// class DerivedOther : public BaseOther {
//   public:
//     void print_type(const BaseDataset &base_dataset) override final {
//       base_dataset.print_type();
//     };
// };

// class DerivedAnother: public BaseOther {
//   public:
//     DerivedAnother(const std::shared_ptr<BaseDataset> &dataset) : dataset(dataset) {}

//     std::shared_ptr<BaseDataset> dataset;

//     virtual void print_type(const BaseDataset &dataset) override final {
//       std::cout << "Typeid name of the input dataset: " << typeid(dataset).name() << std::endl;
//       std::cout << "Typeid name of the desired dataset: " << typeid(LabelledDataset< int, char >).name() << std::endl;
//       std::cout << "Checking if the Dataset type is LabelledDataset< double, char >: " << (typeid(dataset) == typeid(LabelledDataset< int, char >)) << std::endl;
//     }


// };

template<typename T, typename U = int>
struct Foo{
  Foo() {
    std::cout << "Helloooooo" << std::endl;
  }
};

template<typename T>
struct Foo<T, int> {
  Foo(){
    std::cout << "It is an int nowwwwww" << std::endl;
  }
};


int main()
{
  std::cout << "Is it a vector? " << tudat_learn::is_vector<int>::value << std::endl;
  std::cout << "Is it a vector? " << tudat_learn::is_vector<std::vector<std::pair<int,double>>>::value << std::endl;


  Foo<double, double> f;
  Foo<double, int> f1;
  Foo<double> f3;
  

  // UnlabelledDataset un_d(std::vector({1, 2, 3}));
  // un_d.print_type();
  tudat_learn::Dataset l_d(std::vector<int>({1, 2, 3}), std::vector<char>({'a', 'b', 'c'}));
  tudat_learn::Dataset<int> l_d1(std::vector<int>({1, 2, 3}));
  // l_d.print_type();

  auto ldptr = std::make_shared< tudat_learn::Dataset<int, char> >(l_d);

  // tudat_learn::RBFN<int, char> rbfn_instance(ldptr);

  // using type_test = Eigen::Matrix<float, 5, 1>;
  // using type_test = Eigen::VectorXf;

  // type_test vec1(5); vec1 << 1.0,2.0,3.0,2.0,1.0;
  // type_test vec2(5); vec2 << 1.0,2.0,3.0,2.0,1.0;

  // tudat_learn::Dataset new_dataset(std::vector<type_test>({vec1, vec2}), std::vector<char>({'a', 'b', 'c'}));
  // auto new_ptr = std::make_shared< tudat_learn::Dataset<type_test, char> >(new_dataset);

  // tudat_learn::RBFN<type_test, char> rbfn_instance(new_ptr);

  using type_test = Eigen::VectorXf;

  type_test vec1(5); vec1 << 1.0,2.0,3.0,2.0,1.0;
  type_test vec2(5); vec2 << 1.0,2.0,3.0,2.0,1.0;

  tudat_learn::Dataset new_dataset(std::vector<type_test>({vec1, vec2}), std::vector<float>({1, 2, 3}));
  auto new_ptr = std::make_shared< tudat_learn::Dataset<type_test, float> >(new_dataset);

  tudat_learn::RBFN<type_test, float> rbfn_instance(new_ptr);

  // std::vector<

  // DerivedOther other;
  // other.print_type(un_d);
  // other.print_type(l_d);

  // l_d.print_data();
  // auto ptr_l_d = std::shared_ptr< LabelledDataset<int, char> >(&l_d);
  // l_d.data[0] = 5;
  // l_d.print_data();
  // ptr_l_d->print_data();
  // DerivedAnother another(ptr_l_d);
  
  // another.print_type(l_d);

  std::cout << "Hello World." << std::endl;
  Eigen::VectorXd      ev1(100), ev2(100), ev3(100);
  

  for(int i = 0; i < 100; ++i) {
    ev1(i) = rand() / RAND_MAX;
    ev2(i) = rand() / RAND_MAX;
    // sv1[i] = rand() / RAND_MAX;
    // sv2[i] = rand() / RAND_MAX;
  }

  Timer t;
  t.start();
  for(int i = 0; i < 100000; ++i) {
    ev3 = ev1 + ev2;
  }
  t.stop();
  std::cout << "Adding Eigen::Vectors takes " << t.seconds() << " seconds." << std::endl;

  

  return 0;
}