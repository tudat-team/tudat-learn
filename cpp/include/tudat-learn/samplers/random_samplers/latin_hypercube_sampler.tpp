#ifndef TUDAT_LEARN_LATIN_HYPERCUBE_SAMPLER_TPP
#define TUDAT_LEARN_LATIN_HYPERCUBE_SAMPLER_TPP

#ifndef TUDAT_LEARN_LATIN_HYPERCUBE_SAMPLER_HPP
#ERROR __FILE__ should only be included from random_samplers/latin_hypercube_sampler.hpp
#endif

namespace tudat_learn
{

template <typename Datum_t>
std::vector<Datum_t> LatinHypercubeSampler<Datum_t>::sample( ) const {
  std::vector<int> single_indices_vector(buckets_per_dimension);
  std::iota(single_indices_vector.begin(), single_indices_vector.end(), 0);

  int dimensions = this->get_dimensions(this->range);

  // contains vectors with buckets_per_dimension elements which are vectors themselves
  // each of those vectors is filled in with 0, ..., buckets_per_dimension - 1
  // std::vector< std::vector<int> > ordered_indices(dimensions, single_indices_vector);

  // // contains the vectors with the bucket indices in order for each sample
  // std::vector< std::vector<int> > sampled_indices(dimensions, std::vector<int>(buckets_per_dimension));

  // for(int b = 0; b < buckets_per_dimension; ++b) {
  //   for(int d = 0; d < dimensions; ++d) {
  //     std::uniform_int_distribution<int> uniform(0, buckets_per_dimension - 1 - b);
  //     int r = uniform(Random::get_rng()); // choose an index between 0 and #(remaining buckets)

  //     // putting the sampled index on the corresponding vector
  //     sampled_indices.at(d).at(b) = ordered_indices.at(d).at(r);
      
  //     // putting the last eligible element in the place of the one that was just sampled.
  //     ordered_indices.at(d).at(r) = ordered_indices.at(d).at(buckets_per_dimension - 1 - b);
  //   }
  // }

  std::vector< std::vector<int> > sampled_indices(dimensions, single_indices_vector);
  for(int d = 0; d < dimensions; ++d)
    std::shuffle(sampled_indices.at(d).begin(), sampled_indices.at(d).end(), Random::get_rng());

  for(const auto &it : sampled_indices) {
      for(const auto &itt: it)
        std::cout << itt << ", ";
      std::cout << "\n" << std::endl;
    }

  std::vector<Datum_t> selected_buckets = this->generate_buckets(sampled_indices);
  this->print_vector_datum_t(selected_buckets);

  std::vector<Datum_t> samples;
  samples.reserve(buckets_per_dimension);
  for(int b = 0; b < selected_buckets.size(); ++b) {
    // generate random datum between 0 and 1
    Datum_t new_sample(this->sample_zero_one());

    new_sample = this->operator_elementwise_multiplication(new_sample, bucket_size);

    selected_buckets.at(b) = this->operator_elementwise_multiplication(selected_buckets.at(b), bucket_size);

    new_sample = this->operator_addition(new_sample, selected_buckets.at(b));

    new_sample = this->operator_addition(new_sample, this->range.first);
    
    samples.push_back(new_sample);


    

    // multiply/add by buckets
  }
  std::cout << "Bucket size:" << std::endl;
  this->print_datum_t(bucket_size);
  this->print_vector_datum_t(samples);

  return samples;
}

template <typename Datum_t>
std::vector<Datum_t> LatinHypercubeSampler<Datum_t>::sample(const std::pair<Datum_t, Datum_t> &new_range, const int number_samples) {
  this->set_range(new_range);
  set_buckets(number_samples);

  return sample();
}

template <typename Datum_t> template <typename Datum_tt>
typename std::enable_if< std::is_arithmetic<Datum_tt>::value,
std::vector<Datum_tt> >::type LatinHypercubeSampler<Datum_t>::generate_buckets(const std::vector<std::vector<int>> &sampled_indices) const {
  return sampled_indices.at(0);
}


template <typename Datum_t> template <typename Datum_tt>
typename std::enable_if<           is_eigen<Datum_tt>::value,
std::vector<Datum_tt> >::type LatinHypercubeSampler<Datum_t>::generate_buckets(const std::vector<std::vector<int>> &sampled_indices) const {
  std::vector<Datum_tt> buckets;
  buckets.reserve(buckets_per_dimension);

  for(int b = 0; b < buckets_per_dimension; ++b) {
    Datum_tt indices = Datum_tt::NullaryExpr(this->range.first.rows(), this->range.first.cols(),
      [&](Eigen::Index i, Eigen::Index j){ return sampled_indices.at(i * this->range.first.cols() + j).at(b); });

    buckets.push_back(indices);
  }

  return buckets;
}

template <typename Datum_t> template <typename Datum_tt>
typename std::enable_if< is_stl_vector<Datum_tt>::value && std::is_arithmetic<typename Datum_tt::value_type>::value,
std::vector<Datum_tt> >::type LatinHypercubeSampler<Datum_t>::generate_buckets(const std::vector<std::vector<int>> &sampled_indices) const {
  std::vector<Datum_tt> buckets;
  buckets.reserve(buckets_per_dimension);

  for(int b = 0; b < buckets_per_dimension; ++b) {
    Datum_tt indices;
    indices.reserve(sampled_indices.size());

    for(int d = 0; d < sampled_indices.size(); ++d)
      indices.push_back(sampled_indices.at(d).at(b));

    buckets.push_back(indices);
  }

  return buckets;
}

} // namespace tudat_learn

#endif // TUDAT_LEARN_LATIN_HYPERCUBE_SAMPLER_TPP