#ifndef BOOST_FUNCTIONAL_HASH_HPP_
#define BOOST_FUNCTIONAL_HASH_HPP_

#include <cstddef>
#include <functional>
#include <iterator>

namespace boost {

template <typename T>
inline void hash_combine(std::size_t& seed, const T& value) {
  seed ^= std::hash<T>{}(value) + 0x9e3779b97f4a7c15ULL + (seed << 6) +
          (seed >> 2);
}

template <typename It>
inline std::size_t hash_range(It first, It last) {
  std::size_t seed = 0;
  for (; first != last; ++first) {
    hash_combine(seed, *first);
  }
  return seed;
}

}  // namespace boost

#endif  // BOOST_FUNCTIONAL_HASH_HPP_
