#include <iostream>
#include <memory>
#include <typeinfo>
#include <vector>

#include <Eigen/Core>

#include "tudat-learn/samplers/random_samplers/latin_hypercube_sampler.hpp"


  
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




int main()
{

  for(int i = 0; i < 10; ++i) {
    std::uniform_int_distribution<int> dis(0,10);
    std::mt19937 rng(0);
    std::cout << dis(rng) << std::endl;
  }

  int repetitions = 100000;

  tudat_learn::LatinHypercubeSampler<Eigen::Vector4f> lhs(
    std::make_pair(Eigen::Vector4f(1,2,3,4), Eigen::Vector4f(3,4,5,6)),
    5
  );
  
  Timer t;
  


  t.start();
  for(int i = 0; i < repetitions; ++i)
    lhs.sample();
  t.stop();
  std::cout << "LHS Sample takes " << t.seconds() << " seconds." << std::endl;


  // Timer t;
  // t.start();
  // for(int i = 0; i < 100000; ++i) {
  //   ev3 = ev1 + ev2;
  // }
  // t.stop();
  // std::cout << "Adding Eigen::Vectors takes " << t.seconds() << " seconds." << std::endl;

  

  return 0;
}