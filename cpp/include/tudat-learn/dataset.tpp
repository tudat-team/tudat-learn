/*    Copyright (c) 2010-2018, Delft University of Technology
 *    All rights reserved
 *
 *    This file is part of the Tudat. Redistribution and use in source and
 *    binary forms, with or without modification, are permitted exclusively
 *    under the terms of the Modified BSD license. You should have received
 *    a copy of the license with this file. If not, please or visit:
 *    http://tudat.tudelft.nl/LICENSE.
 */

#ifndef TUDAT_LEARN_DATASET_TPP
#define TUDAT_LEARN_DATASET_TPP

#ifndef TUDAT_LEARN_DATASET_HPP
#ERROR __FILE__ should only be included from dataset.hpp
#endif

namespace tudat_learn
{
  template <typename Datum_t, typename Label_t>
template <typename Datum_tt>
typename std::enable_if< is_floating_point_eigen_vector<Datum_tt>::value, std::vector<int> >::type
Dataset<Datum_t, Label_t>::get_closest_data(Datum_tt vector_of_interest, int amount) { 

  std::vector<int> all_indices(data.size());
  std::iota(all_indices.begin(), all_indices.end(), 0); // fills each element with the respective index: v[i] = i
  if(amount == -1 || amount >= data.size())
    return all_indices;

  // Creates a vector of the same scalar type held by the Eigen::Matrix in Datum_tt
  std::vector<typename Datum_tt::Scalar> radii;
  radii.reserve(data.size());

  // Computes the radius between the `vector_of_interest` and every instance in the dataset.
  for(const auto &it : data)
    radii.push_back((it - vector_of_interest).norm());

  // sorts the `amount` indices in the `all_indices` vector with the corresponding largest radius from `radii`
  std::partial_sort(all_indices.begin(), all_indices.begin() + amount, all_indices.end(), 
                    [&] (int i, int j) { return radii[i] < radii[j]; } 
  );

  std::vector<int> return_indices(all_indices.begin(), all_indices.begin() + amount);

  return return_indices;  
}

template <typename Datum_t>
template <typename Datum_tt>
typename std::enable_if< is_floating_point_eigen_vector<Datum_tt>::value, std::vector<int> >::type
Dataset<Datum_t>::get_closest_data(Datum_tt vector_of_interest, int amount) { 

  std::vector<int> all_indices(data.size());
  std::iota(all_indices.begin(), all_indices.end(), 0); // fills each element with the respective index: v[i] = i
  if(amount == -1 || amount >= data.size())
    return all_indices;

  // Creates a vector of the same scalar type held by the Eigen::Matrix in Datum_tt
  std::vector<typename Datum_tt::Scalar> radii;
  radii.reserve(data.size());

  // Computes the radius between the `vector_of_interest` and every instance in the dataset.
  for(const auto &it : data)
    radii.push_back((it - vector_of_interest).norm());

  // sorts the `amount` indices in the `all_indices` vector with the corresponding largest radius from `radii`
  std::partial_sort(all_indices.begin(), all_indices.begin() + amount, all_indices.end(), 
                    [&] (int i, int j) { return radii[i] < radii[j]; } 
  );

  std::vector<int> return_indices(all_indices.begin(), all_indices.begin() + amount);

  return return_indices;  
}
} // namespace tudat_learn

#endif // TUDAT_LEARN_DATASET_TPP