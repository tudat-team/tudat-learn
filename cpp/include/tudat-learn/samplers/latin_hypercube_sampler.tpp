#ifndef TUDAT_LEARN_LATIN_HYPERCUBE_SAMPLER_TPP
#define TUDAT_LEARN_LATIN_HYPERCUBE_SAMPLER_TPP

#ifndef TUDAT_LEARN_LATIN_HYPERCUBE_SAMPLER_HPP
#ERROR __FILE__ should only be included from samplers/latin_hypercube_sampler.hpp
#endif

namespace tudat_learn
{

template <typename Datum_t>
std::vector<Datum_t> LatinHypercubeSampler<Datum_t>::sample( ) const {
  std::vector<int> single_indices_vector(buckets_per_dimension);
  std::iota(single_indices_vector.begin(), single_indices_vector.end(), 0);

  // contains vectors with buckets_per_dimension elements which are vectors themselves
  // each of those vectors is filled in with 0, ..., buckets_per_dimension - 1
  // each vector corresponds to a dimension and each index corresponds to a bucket
  // std::vector< std::vector<int> > indices_vector_vector(buckets_per_dimension, single_indices_vector);

  // std::vector< std::vector<int>
  // WRONGGGG
}

template <typename Datum_t>
std::vector<Datum_t> LatinHypercubeSampler<Datum_t>::sample(const std::pair<Datum_t, Datum_t> &new_range, const int new_buckets_per_dimension) {
  set_range(new_range);
  set_buckets(new_buckets_per_dimension);

  return sample();
}

} // namespace tudat_learn

#endif // TUDAT_LEARN_LATIN_HYPERCUBE_SAMPLER_TPP