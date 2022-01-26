# /*    Copyright (c) 2010-2018, Delft University of Technology
#  *    All rights reserved
#  *
#  *    This file is part of the Tudat. Redistribution and use in source and
#  *    binary forms, with or without modification, are permitted exclusively
#  *    under the terms of the Modified BSD license. You should have received
#  *    a copy of the license with this file. If not, please or visit:
#  *    http://tudat.tudelft.nl/LICENSE.
#  */

import numpy as np

if __name__ == '__main__':
  np.random.seed(0)

  range = [0, 1]

  data = np.around(np.random.rand(10, 7), 6)
  print("Data:")
  print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in data))

  minimum = np.amin(data, axis=0)
  print("Minimum:")
  print(', '.join(str(format(n, '.6f')) for n in minimum))

  maximum = np.amax(data, axis=0)
  print("Maximum:")
  print(', '.join(str(format(n, '.6f')) for n in maximum))

  scaled_data = (data - minimum) / (maximum - minimum) * (range[1] - range[0]) + range[0]
  print("Scaled Data:")
  print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in scaled_data))

  # scalars
  data_scalar = np.around(np.random.rand(10), 6)
  print("Data Scalar:")
  print(', '.join(str(format(n, '.6f')) for n in data_scalar))

  minimum_scalar = np.around(np.amin(data_scalar), 6)
  print("Minimum Scalar:")
  print(minimum_scalar)

  maximum_scalar = np.around(np.amax(data_scalar), 6)
  print("Maximum Scalar:")
  print(maximum_scalar)

  scaled_data_scalar = (data_scalar - minimum_scalar) / (maximum_scalar - minimum_scalar) * (range[1] - range[0]) + range[0]
  print("Scaled Data Scalar:")
  print(', '.join(str(format(n, '.6f')) for n in scaled_data_scalar))

  # matrices
  data_matrix = np.around(np.random.rand(4,2,2), 6)
  print("Data Matrix:")
  print(',\n\n'.join((',\n'.join((', '.join(str(format(nnn, '.6f')) for nnn in nn)) for nn in n)) for n in data_matrix))

  minimum_matrix = np.around(np.amin(data_matrix, axis=0), 6)
  print("Minimum Matrix:")
  print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in minimum_matrix))

  maximum_matrix = np.around(np.amax(data_matrix, axis=0), 6)
  print("Maximum Matrix:")
  print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in maximum_matrix))

  scaled_data_matrix = (data_matrix - minimum_matrix) / (maximum_matrix - minimum_matrix) * (range[1] - range[0]) + range[0]
  print("Scaled Data Matrix:")
  print(',\n\n'.join((',\n'.join((', '.join(str(format(nnn, '.6f')) for nnn in nn)) for nn in n)) for n in scaled_data_matrix))

  # different range
  range_new = [-5, 7]

  minimum_range = np.amin(data, axis=0)
  print("Minimum Range:")
  print(', '.join(str(format(n, '.6f')) for n in minimum_range))

  maximum_range = np.amax(data, axis=0)
  print("Maximum Range:")
  print(', '.join(str(format(n, '.6f')) for n in maximum_range))

  scaled_data_range = (data - minimum_range) / (maximum_range - minimum_range) * (range_new[1] - range_new[0]) + range_new[0]
  print("Scaled Data Range:")
  print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in scaled_data_range))