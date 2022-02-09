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

  data = np.around(np.random.rand(10, 7), 6)
  print("Data:")
  print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in data))

  mean = np.mean(data, axis=0)
  print("Mean:")
  print(', '.join(str(format(n, '.6f')) for n in mean))

  standard_deviation = np.std(data, axis=0)
  print("Standard Deviation:")
  print(', '.join(str(format(n, '.6f')) for n in standard_deviation))

  variance = np.var(data, axis=0)
  print("Variance:")
  print(', '.join(str(format(n, '.6f')) for n in variance))

  scaled_data = (data - mean) / standard_deviation
  print("Scaled Data:")
  print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in scaled_data))

  # scalars
  data_scalar = np.around(np.random.rand(10), 6)
  print("Data Scalar:")
  print(', '.join(str(format(n, '.6f')) for n in data_scalar))

  mean_scalar = np.around(np.mean(data_scalar), 6)
  print("Mean Scalar Data:")
  print(mean_scalar)

  std_scalar = np.around(np.std(data_scalar), 6)
  print("Standard Deviation Scalar Data:")
  print(std_scalar)

  var_scalar = np.around(np.var(data_scalar), 6)
  print("Variance Scalar Data:")
  print(var_scalar)

  scaled_data_scalar = (data_scalar - mean_scalar) / std_scalar
  print("Scaled Data Scalar:")
  print(', '.join(str(format(n, '.6f')) for n in scaled_data_scalar))

  # matrices
  data_matrix = np.around(np.random.rand(4,2,2), 6)
  print("Data Matrix:")
  print(',\n\n'.join((',\n'.join((', '.join(str(format(nnn, '.6f')) for nnn in nn)) for nn in n)) for n in data_matrix))

  mean_matrix = np.around(np.mean(data_matrix, axis=0), 6)
  print("Mean Matrix Data:")
  print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in mean_matrix))

  std_matrix = np.around(np.std(data_matrix, axis=0), 6)
  print("Standard Deviation Matrix Data:")
  print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in std_matrix))

  var_matrix = np.around(np.var(data_matrix, axis=0), 6)
  print("Variance Matrix Data:")
  print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in var_matrix))

  scaled_data_matrix = (data_matrix - mean_matrix) / std_matrix
  print("Scaled Data Matrix:")
  print(',\n\n'.join((',\n'.join((', '.join(str(format(nnn, '.6f')) for nnn in nn)) for nn in n)) for n in scaled_data_matrix))