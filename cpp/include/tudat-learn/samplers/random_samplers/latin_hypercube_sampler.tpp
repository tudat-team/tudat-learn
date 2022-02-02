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

  // contains vectors with buckets_per_dimension elements which are vectors themselves
  // each of those vectors is filled in with 0, ..., buckets_per_dimension - 1
  std::vector< std::vector<int> > ordered_indices(get_dimensions(this->range), single_indices_vector);

  // contains the vectors with the bucket indices in order for each sample
  std::vector< std::vector<int> > sampled_indices(get_dimensions(this->range), std::vector<int>(buckets_per_dimension));

  // for(int i = 0; i < orde)
}

template <typename Datum_t>
std::vector<Datum_t> LatinHypercubeSampler<Datum_t>::sample(const std::pair<Datum_t, Datum_t> &new_range, const int number_samples) {
  this->set_range(new_range);
  set_buckets(number_samples);

  return sample();
}

} // namespace tudat_learn

#endif // TUDAT_LEARN_LATIN_HYPERCUBE_SAMPLER_TPP