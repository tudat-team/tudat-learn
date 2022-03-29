# /*    Copyright (c) 2010-2018, Delft University of Technology
#  *    All rights reserved
#  *
#  *    This file is part of the Tudat. Redistribution and use in source and
#  *    binary forms, with or without modification, are permitted exclusively
#  *    under the terms of the Modified BSD license. You should have received
#  *    a copy of the license with this file. If not, please or visit:
#  *    http://tudat.tudelft.nl/LICENSE.
#  */

from random import gauss
from traceback import print_tb
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error

import numpy as np

if __name__ == '__main__':
    np.random.seed(0)
    # precision = 10

    center_points = np.around(np.random.rand(10, 7), 6)
    print("Center Points:")
    print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in center_points))

    output = np.around(np.random.rand(10, 2), 6)
    print("Output:")
    print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in output))

    distance_matrix = np.sqrt(np.sum((center_points[:, np.newaxis, :] - center_points[np.newaxis, :, :]) **2, axis=-1))
    # print("Distance Matrix:")
    # print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in distance_matrix))

    # computing parameters (except sigma)

    sigma = np.around(np.random.rand(1), 6)
    print("Sigma is", str(format(sigma[0], '.6f')))

    gaussian_distance_matrix = np.exp(- (distance_matrix / sigma)**2 )
    # print("Gaussian Distance Matrix:")
    # print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in gaussian_distance_matrix))

    cv_splits = [
      [[3,4,5,6,7,8,9], [0,1,2]],
      [[0,1,2,6,7,8,9], [3,4,5]],
      [[0,1,2,3,4,5], [6,7,8,9]]
    ]

    # dm = np.zeros((7, 7))
    # for i in range(7):
    #     for j in range(7):
    #         dm[i, j] = np.linalg.norm(center_points[cv_splits[0][0][i],:] - center_points[cv_splits[0][0][j], :])
    # print(dm)
    
    train_distance_matrices = []
    test_distance_matrices = []
    gaussian_coefficients = []

    for i in range(len(cv_splits)):
      current_distance_matrix = np.sqrt(np.sum((center_points[cv_splits[i][0], np.newaxis, :] - center_points[np.newaxis, cv_splits[i][0], :]) **2, axis=-1))
      current_gaussian_distance_matrix = np.exp(- (current_distance_matrix / sigma)**2 )
      print("Gaussian Distance Matrix", i, ":")  
      print(',\n'.join((', '.join(str(format(nn, '.6f')) for nn in n)) for n in current_gaussian_distance_matrix))

      train_distance_matrices.append(current_gaussian_distance_matrix)
      gaussian_coefficients.append(np.linalg.solve(current_gaussian_distance_matrix, output[cv_splits[i][0]]))

      current_distance_matrix_test = np.sqrt(np.sum((center_points[cv_splits[i][1], np.newaxis, :] - center_points[np.newaxis, cv_splits[i][0], :]) **2, axis=-1))
      current_gaussian_distance_matrix_test = np.exp(- (current_distance_matrix_test / sigma)**2 )
      test_distance_matrices.append(current_gaussian_distance_matrix_test)

    fold_metrics = []
    for i in range(len(cv_splits)):
      metrics = []
      expected_output = np.matmul(test_distance_matrices[i], gaussian_coefficients[i])
      metrics.append(mean_absolute_percentage_error(output[cv_splits[i][1]], expected_output))
      metrics.append(mean_absolute_error(output[cv_splits[i][1]], expected_output))
      fold_metrics.append(metrics)

    for i, metrics_in_fold in enumerate(fold_metrics):
      print("Metrics in fold", i, ":")
      print(', '.join(str(n) for n in metrics_in_fold))
      # print(', '.join(str(format(n, '.6f')) for n in metrics_in_fold))
