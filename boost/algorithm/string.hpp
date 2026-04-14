#ifndef BOOST_ALGORITHM_STRING_HPP_
#define BOOST_ALGORITHM_STRING_HPP_

#include <algorithm>
#include <cctype>
#include <string>

namespace boost {

enum token_compress_mode_type {
  token_compress_off,
  token_compress_on,
};

class is_any_of {
 public:
  explicit is_any_of(std::string delimiters)
      : delimiters_(std::move(delimiters)) {}

  bool operator()(char ch) const {
    return delimiters_.find(ch) != std::string::npos;
  }

 private:
  std::string delimiters_;
};

inline void trim(std::string& value) {
  auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
  auto first = std::find_if(value.begin(), value.end(), not_space);
  auto last = std::find_if(value.rbegin(), value.rend(), not_space).base();
  if (first >= last) {
    value.clear();
    return;
  }
  value.assign(first, last);
}

inline void to_lower(std::string& value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
}

template <typename SequenceSequenceT, typename PredicateT>
void split(SequenceSequenceT& output, const std::string& input,
           PredicateT is_delimiter,
           token_compress_mode_type compress_mode = token_compress_off) {
  output.clear();

  std::string current;
  for (char ch : input) {
    if (is_delimiter(ch)) {
      if (compress_mode == token_compress_on) {
        if (!current.empty()) {
          output.push_back(current);
          current.clear();
        }
      } else {
        output.push_back(current);
        current.clear();
      }
      continue;
    }
    current.push_back(ch);
  }

  if (!current.empty() || compress_mode == token_compress_off) {
    output.push_back(current);
  }
}

}  // namespace boost

#endif  // BOOST_ALGORITHM_STRING_HPP_
