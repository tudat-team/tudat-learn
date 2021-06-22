/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_RASTRIGIN_H
#define TUDAT_LEARN_RASTRIGIN_H
#include <tudat/learn/function.h>

typedef std::pair<std::vector<double>, std::vector<double>> pair_vector_double;

class Rastrigin : public Function {
public:
  //! Rastrigin function
  /// f(\mathbf {x} )=An+\sum _{i=1}^{n}\left[x_{i}^{2}-A\cos(2\pi x_{i})\right]
  /// \param dim
  Rastrigin(int n, double a = 10.) {
    this->_n = n;
    this->_a = a;
  }

  double f(std::vector<double> x) { // TODO: Should this be a vector? Perhaps.
    // TODO: should assert x.len() == n
    double f = 0;
    for (double x_i : x) {
      f += x_i * x_i - _a * cos(2 * PI * x_i)
    }
    return f + _a * _n;
  }

  int get_n() { return _n; }

  pair_vector_double get_bounds() {
    std::vector<double> lower;
    std::vector<double> upper;
    for (i = 0; i < _n; i++) {
      lower.push(-5.12);
      upper.push(+5.12);
    }
  };

  pair_vector_double get_optima() {
    std::vector<std::vector<double>> optima;
    std::vector<double> optimum;
    for (i = 0; i < this->_n; i++) {
      optimum.push(0.);
    }
    optima.push(optimum);
    return optima;
  };

private:
  int _n;
  int _bounds;
  double _a;
};

#endif // TUDAT_LEARN_RASTRIGIN_H
