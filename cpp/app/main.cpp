#include <iostream>
#include <vector>

#include <Eigen/Core>

#include "dataset.h"
#include "estimators/regressors/rbfn.h"


  
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




int main()
{
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

  t.start();
  std::vector<double>  sv1(100), sv2(100), sv3(100);
   for(int i = 0; i < 100; ++i) {
    sv1[i] = rand() / RAND_MAX;
    sv2[i] = rand() / RAND_MAX;
  }
  for(int i = 0; i < 100000; ++i) {
    for(int j = 0; j < 100; ++j)
      sv3[j] = sv1[j] + sv2[j];
  }
  t.stop();
  std::cout << "Adding std::vectors takes " << t.seconds() << " seconds." << std::endl;

  // tudat_learn::Dataset d;
  // tudat_learn::RBFN rbfn;

  // rbfn.fit(d);

  t.start();
  for(int i = 0; i < 10000; ++i){
    {
      std::vector<double> vec1(10000);
      for(int j = 0; j < 10000; ++j) {
        vec1[j] = rand() / RAND_MAX;
      }
    }
  }
  t.stop();
  std::cout << "Initialisation with size takes " << t.seconds() << " seconds." << std::endl;

  t.start();
  for(int i = 0; i < 10000; ++i){
    {
      std::vector<double> vec1;
      vec1.reserve(10000);
      for(int j = 0; j < 10000; ++j) {
        vec1[j] = rand() / RAND_MAX;
      }
    }
  }
  t.stop();
  std::cout << "Default initialisation + reserve with size takes " << t.seconds() << " seconds." << std::endl;

  return 0;
}