# /*    Copyright (c) 2010-2018, Delft University of Technology
#  *    All rights reserved
#  *
#  *    This file is part of the Tudat. Redistribution and use in source and
#  *    binary forms, with or without modification, are permitted exclusively
#  *    under the terms of the Modified BSD license. You should have received
#  *    a copy of the license with this file. If not, please or visit:
#  *    http://tudat.tudelft.nl/LICENSE.
#  */

import random

from sympy import *
import numpy as np


if __name__ == '__main__':
  x1 = Symbol('x1')
  x2 = Symbol('x2')
  x3 = Symbol('x3')
  c1 = Symbol('c1')
  c2 = Symbol('c2')
  c3 = Symbol('c3')
  sigma = Symbol('sigma')

  random.seed(0)
  x = np.array([random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)])
  print("x is ", x)

  c = np.array([random.uniform(0,1), random.uniform(0,1), random.uniform(0,1)])
  print("c is ", c)

  sigma_val = random.uniform(0,1)
  print("sigma is ", sigma_val)

  # Cubic
  cubic = Matrix([((x1 - c1)**2 + (x2 - c2)**2 + (x3 - c3)**2)**(3/2)])
  cubic_l = lambdify((x1, x2, x3, c1, c2, c3), cubic, 'numpy')
  print("Cubic evaluated at x and c is: ", cubic_l(x[0], x[1], x[2], c[0], c[1], c[2]))

  cubic_gradient = cubic.gradient([x1, x2, x3])
  cubic_gradient_l = lambdify((x1, x2, x3, c1, c2, c3), cubic_gradient, 'numpy')
  print("Cubic gradient evaluated at x and c is: ", cubic_gradient_l(x[0], x[1], x[2], c[0], c[1], c[2]))

  cubic_hessian = hessian(cubic, (x1, x2, x3))
  cubic_hessian_l = lambdify((x1, x2, x3, c1, c2, c3), cubic_hessian, 'numpy')
  print("Cubic Hessian evaluated at x and c is:\n", cubic_hessian_l(x[0], x[1], x[2], c[0], c[1], c[2]))

  # Gaussian
  gaussian = Matrix([exp(- ((x1 - c1)**2 + (x2 - c2)**2 + (x3 - c3)**2) / sigma**2)])
  gaussian_l = lambdify((x1, x2, x3, c1, c2, c3, sigma), gaussian, 'numpy')
  print("Gaussian evaluated at x, c and sigma is: ", gaussian_l(x[0], x[1], x[2], c[0], c[1], c[2], sigma_val))

  gaussian_gradient = gaussian.gradient([x1, x2, x3])
  gaussian_gradient_l = lambdify((x1, x2, x3, c1, c2, c3, sigma), gaussian_gradient, 'numpy')
  print("Gaussian gradient evaluated at x, c and sigma is: ", gaussian_gradient_l(x[0], x[1], x[2], c[0], c[1], c[2], sigma_val))

  gaussian_hessian = hessian(gaussian, (x1, x2, x3))
  gaussian_hessian_l = lambdify((x1, x2, x3, c1, c2, c3, sigma), gaussian_hessian, 'numpy')
  print("Gaussian Hessian evaluated at x, c and sigma is:\n", gaussian_hessian_l(x[0], x[1], x[2], c[0], c[1], c[2], sigma_val))

  #Expressions
  print("\nCubic:\n", cubic)

  print("\nCubic gradient:\n", cubic_gradient)

  print("\nCubic Hessian:\n")
  for line in cubic_hessian:
    print(line)

  print("\nGaussian:\n", gaussian)

  print("\nGaussian gradient:\n", gaussian_gradient)

  print("\nGaussian Hessian:\n")
  for line in gaussian_hessian:
    print(line)
  