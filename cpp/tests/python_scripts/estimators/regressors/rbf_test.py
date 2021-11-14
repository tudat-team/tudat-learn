from sympy import *
import numpy as np

if __name__ == '__main__':
  sigma = Symbol('sigma')
  x1 = Symbol('x1')
  x2 = Symbol('x2')
  x3 = Symbol('x3')
  c1 = Symbol('c1')
  c2 = Symbol('c2')
  c3 = Symbol('c3')
  gaussian = Matrix([exp(- ((x1 - c1)**2 + (x2 - c2)**2 + (x3 - c3)**2) / sigma**2)])

  cubic = Matrix([((x1 - c1)**2 + (x2 - c2)**2 + (x3 - c3)**2)**(3/2)])

  print(gaussian.jacobian([x1, x2, x3]))

  print(cubic.jacobian([x1, x2, x3]))

  print(hessian(cubic, (x1, x2, x3)))

  print()

  for line in hessian(cubic, (x1, x2, x3)):
    print(line)

  print()

  for line in hessian(gaussian, (x1, x2, x3)):
    print(line)