#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cstdint>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <queue>
#include <random>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pushworld {

using Bits = std::vector<std::uint64_t>;
static constexpr int INF = std::numeric_limits<int>::max() / 4;
static constexpr std::array<std::pair<int, int>, 4> DIRS{{{0, -1}, {0, 1}, {-1, 0}, {1, 0}}};

inline std::size_t hash_combine(std::size_t seed, std::size_t value) {
    return seed ^ (value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2));
}

struct VecHash {
    std::size_t operator()(const std::vector<int>& v) const noexcept {
        std::size_t h = 0;
        for (int x : v) h = hash_combine(h, std::hash<int>{}(x));
        return h;
    }
};

struct Polyomino {
    std::string name;
    std::vector<std::pair<int, int>> cells;
    bool has_target = false;

    [[nodiscard]] Polyomino normalized() const {
        int min_x = std::numeric_limits<int>::max();
        int min_y = std::numeric_limits<int>::max();
        for (const auto& [x, y] : cells) {
            min_x = std::min(min_x, x);
            min_y = std::min(min_y, y);
        }
        std::vector<std::pair<int, int>> out;
        out.reserve(cells.size());
        for (const auto& [x, y] : cells) out.emplace_back(x - min_x, y - min_y);
        std::sort(out.begin(), out.end());
        return Polyomino{name, out, has_target};
    }

    [[nodiscard]] int width() const {
        int w = 0;
        for (const auto& [x, _] : cells) w = std::max(w, x + 1);
        return w;
    }

    [[nodiscard]] int height() const {
        int h = 0;
        for (const auto& [_, y] : cells) h = std::max(h, y + 1);
        return h;
    }
};

struct BoardSpec {
    int width = 0;
    int height = 0;
    Bits walls;
    Polyomino player;
    std::vector<Polyomino> boxes;
    std::string profile_scope = "all";
    std::optional<std::uint64_t> wall_seed;
};

struct StartState {
    std::pair<int, int> player_anchor;
    std::vector<std::pair<int, int>> box_anchors;
};

struct HardestPuzzleResult {
    std::vector<int> profile;
    std::vector<int> per_box_min_pushes;
    StartState start_state;
    std::vector<std::optional<std::pair<int, int>>> target_anchors_by_box;
    BoardSpec board;

    [[nodiscard]] std::string render() const {
        std::vector<std::vector<std::string>> grid(
            static_cast<std::size_t>(board.height),
            std::vector<std::string>(static_cast<std::size_t>(board.width), ".."));

        const auto overlay_target = [&](int x, int y, const std::string& target_name) {
            if (grid[y][x] == "..") grid[y][x] = "." + target_name;
        };
        const auto overlay_occupant = [&](int x, int y, const std::string& occupant_name) {
            const char suffix = grid[y][x].size() == 2 ? grid[y][x][1] : '.';
            grid[y][x] = occupant_name + std::string(1, suffix);
        };

        const auto bit_index = [&](int x, int y) { return y * board.width + x; };
        auto bit_test = [&](const Bits& bits, int x, int y) {
            const int idx = bit_index(x, y);
            return ((bits[idx / 64] >> (idx % 64)) & 1ull) != 0ull;
        };

        for (int y = 0; y < board.height; ++y) {
            for (int x = 0; x < board.width; ++x) {
                if (bit_test(board.walls, x, y)) grid[y][x] = "#.";
            }
        }

        for (std::size_t i = 0; i < board.boxes.size(); ++i) {
            if (!board.boxes[i].has_target || !target_anchors_by_box[i].has_value()) continue;
            const auto [ax, ay] = *target_anchors_by_box[i];
            for (const auto& [dx, dy] : board.boxes[i].cells) {
                const int x = ax + dx;
                const int y = ay + dy;
                overlay_target(x, y, board.boxes[i].name);
            }
        }

        for (std::size_t i = 0; i < board.boxes.size(); ++i) {
            const auto [ax, ay] = start_state.box_anchors[i];
            for (const auto& [dx, dy] : board.boxes[i].cells) {
                overlay_occupant(ax + dx, ay + dy, board.boxes[i].name);
            }
        }

        const auto [px, py] = start_state.player_anchor;
        for (const auto& [dx, dy] : board.player.cells) {
            overlay_occupant(px + dx, py + dy, "1");
        }

        std::ostringstream oss;
        for (int y = 0; y < board.height; ++y) {
            for (int x = 0; x < board.width; ++x) {
                if (x) oss << ' ';
                oss << grid[y][x];
            }
            if (y + 1 < board.height) oss << '\n';
        }
        return oss.str();
    }
};

inline Bits make_bits(int area) {
    return Bits(static_cast<std::size_t>((area + 63) / 64), 0ull);
}

inline void set_bit(Bits& bits, int idx) {
    bits[static_cast<std::size_t>(idx / 64)] |= (1ull << (idx % 64));
}

inline bool intersects(const Bits& a, const Bits& b) {
    for (std::size_t i = 0; i < a.size(); ++i) if ((a[i] & b[i]) != 0ull) return true;
    return false;
}

inline Bits bit_or(const Bits& a, const Bits& b) {
    Bits out = a;
    for (std::size_t i = 0; i < out.size(); ++i) out[i] |= b[i];
    return out;
}

inline void or_into(Bits& dst, const Bits& src) {
    for (std::size_t i = 0; i < dst.size(); ++i) dst[i] |= src[i];
}

inline int popcount(const Bits& bits) {
    int total = 0;
    for (auto w : bits) total += std::popcount(w);
    return total;
}

inline Bits generate_walls(int width, int height, int wall_count, std::uint64_t seed) {
    const int area = width * height;
    if (wall_count < 0 || wall_count > area) throw std::invalid_argument("invalid wall_count");
    std::vector<int> cells(area);
    for (int i = 0; i < area; ++i) cells[i] = i;
    std::mt19937_64 rng(seed);
    std::shuffle(cells.begin(), cells.end(), rng);
    Bits walls = make_bits(area);
    for (int i = 0; i < wall_count; ++i) set_bit(walls, cells[i]);
    return walls;
}

class PushWorldEngine {
public:
    explicit PushWorldEngine(BoardSpec spec)
        : spec_(std::move(spec)), width_(spec_.width), height_(spec_.height), area_(width_ * height_) {
        if (width_ <= 0 || height_ <= 0) throw std::invalid_argument("width and height must be positive");
        if (spec_.boxes.empty()) throw std::invalid_argument("at least one box is required");
        spec_.player = spec_.player.normalized();
        for (auto& box : spec_.boxes) box = box.normalized();
        for (int i = 0; i < static_cast<int>(spec_.boxes.size()); ++i) {
            if (spec_.boxes[i].has_target) target_box_indices_.push_back(i);
        }
        if (target_box_indices_.empty()) throw std::invalid_argument("at least one target box is required");
        if (spec_.profile_scope == "targeted") {
            profile_box_indices_ = target_box_indices_;
        } else {
            for (int i = 0; i < static_cast<int>(spec_.boxes.size()); ++i) profile_box_indices_.push_back(i);
        }

        precompute_anchors();
        enumerate_box_configs();
        build_macrostates();
        build_graph();
        build_goal_buckets();
    }

    [[nodiscard]] std::optional<HardestPuzzleResult> find_hardest_puzzle() const {
        std::optional<HardestPuzzleResult> best;
        std::vector<int> best_full_sorted;
        for (const auto& [signature, goal_states] : goal_bucket_by_signature_) {
            std::vector<std::vector<int>> dists_by_box(spec_.boxes.size());
            for (int box = 0; box < static_cast<int>(spec_.boxes.size()); ++box) {
                dists_by_box[box] = reverse_zero_one_bfs(goal_states, box);
            }
            for (int start_macro = 0; start_macro < static_cast<int>(macro_box_cfg_.size()); ++start_macro) {
                if (macro_signature_[start_macro] == signature) continue;
                std::vector<int> per_box(spec_.boxes.size());
                bool reachable = true;
                for (int box = 0; box < static_cast<int>(spec_.boxes.size()); ++box) {
                    const int d = dists_by_box[box][start_macro];
                    if (d >= INF) {
                        reachable = false;
                        break;
                    }
                    per_box[box] = d;
                }
                if (!reachable) continue;
                std::vector<int> profile;
                profile.reserve(profile_box_indices_.size());
                for (int bi : profile_box_indices_) profile.push_back(per_box[bi]);
                std::sort(profile.begin(), profile.end());
                std::vector<int> full_sorted = per_box;
                std::sort(full_sorted.begin(), full_sorted.end());
                if (!best.has_value() || lex_greater(profile, best->profile) ||
                    (profile == best->profile && lex_greater(full_sorted, best_full_sorted))) {
                    HardestPuzzleResult candidate;
                    candidate.profile = profile;
                    candidate.per_box_min_pushes = per_box;
                    candidate.board = spec_;
                    candidate.target_anchors_by_box.resize(spec_.boxes.size());
                    const int cfg_id = macro_box_cfg_[start_macro];
                    const auto& box_anchor_ids = box_cfg_anchors_[cfg_id];
                    const int player_anchor_id = macro_anchors_flat_[macro_anchor_offsets_[start_macro]];
                    candidate.start_state.player_anchor = player_anchor_xy_[player_anchor_id];
                    candidate.start_state.box_anchors.reserve(spec_.boxes.size());
                    for (int i = 0; i < static_cast<int>(spec_.boxes.size()); ++i) {
                        candidate.start_state.box_anchors.push_back(box_anchor_xy_[i][box_anchor_ids[i]]);
                    }
                    const auto& signature_tuple = signature;
                    for (int slot = 0; slot < static_cast<int>(target_box_indices_.size()); ++slot) {
                        const int bi = target_box_indices_[slot];
                        candidate.target_anchors_by_box[bi] = box_anchor_xy_[bi][signature_tuple[slot]];
                    }
                    best = std::move(candidate);
                    best_full_sorted = std::move(full_sorted);
                }
            }
        }
        return best;
    }

private:
    BoardSpec spec_;
    int width_ = 0;
    int height_ = 0;
    int area_ = 0;

    struct AnchorTable {
        std::vector<Bits> occ;
        std::vector<std::pair<int, int>> xy;
        std::unordered_map<std::uint64_t, int> idx_by_xy;
        std::vector<std::array<int, 4>> next;
    };

    AnchorTable player_table_;
    std::vector<AnchorTable> box_tables_;
    std::vector<int> profile_box_indices_;
    std::vector<int> target_box_indices_;

    // Box configurations.
    std::vector<std::vector<int>> box_cfg_anchors_;
    std::vector<Bits> box_cfg_union_;
    std::unordered_map<std::vector<int>, int, VecHash> box_cfg_id_by_anchors_;

    // Macro-states.
    std::vector<int> macro_box_cfg_;
    std::vector<int> macro_anchor_offsets_{0};
    std::vector<int> macro_anchors_flat_;
    std::vector<std::vector<int>> cfg_player_owner_;

    // Reverse graph: reverse_adj_[u] contains (predecessor, push_mask).
    std::vector<std::vector<std::pair<int, std::uint32_t>>> reverse_adj_;

    // Goal buckets keyed by target anchor ids.
    std::unordered_map<std::vector<int>, std::vector<int>, VecHash> goal_bucket_by_signature_;
    std::vector<std::vector<int>> macro_signature_;

    [[nodiscard]] static bool lex_greater(const std::vector<int>& lhs, const std::vector<int>& rhs) {
        return std::lexicographical_compare(rhs.begin(), rhs.end(), lhs.begin(), lhs.end());
    }

    [[nodiscard]] int bit_index(int x, int y) const {
        return y * width_ + x;
    }

    [[nodiscard]] std::uint64_t xy_key(int x, int y) const {
        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(x)) << 32) |
               static_cast<std::uint32_t>(y);
    }

    [[nodiscard]] Bits mask_from_cells(std::span<const std::pair<int, int>> cells) const {
        Bits bits = make_bits(area_);
        for (const auto& [x, y] : cells) set_bit(bits, bit_index(x, y));
        return bits;
    }

    [[nodiscard]] AnchorTable build_anchor_table(const Polyomino& shape) const {
        AnchorTable table;
        for (int y = 0; y <= height_ - shape.height(); ++y) {
            for (int x = 0; x <= width_ - shape.width(); ++x) {
                Bits bits = make_bits(area_);
                bool blocked = false;
                for (const auto& [dx, dy] : shape.cells) {
                    const int xx = x + dx;
                    const int yy = y + dy;
                    set_bit(bits, bit_index(xx, yy));
                }
                if (intersects(bits, spec_.walls)) blocked = true;
                if (!blocked) {
                    const int idx = static_cast<int>(table.occ.size());
                    table.occ.push_back(std::move(bits));
                    table.xy.emplace_back(x, y);
                    table.idx_by_xy.emplace(xy_key(x, y), idx);
                }
            }
        }
        if (table.occ.empty()) throw std::invalid_argument("an object has no legal anchors");
        table.next.reserve(table.xy.size());
        for (const auto& [x, y] : table.xy) {
            std::array<int, 4> nbrs{};
            for (int d = 0; d < 4; ++d) {
                const auto [dx, dy] = DIRS[d];
                auto it = table.idx_by_xy.find(xy_key(x + dx, y + dy));
                nbrs[d] = (it == table.idx_by_xy.end()) ? -1 : it->second;
            }
            table.next.push_back(nbrs);
        }
        return table;
    }

    void precompute_anchors() {
        player_table_ = build_anchor_table(spec_.player);
        box_tables_.clear();
        box_tables_.reserve(spec_.boxes.size());
        for (const auto& box : spec_.boxes) box_tables_.push_back(build_anchor_table(box));
        player_anchor_xy_ = player_table_.xy;
        player_occ_ = player_table_.occ;
        player_next_ = player_table_.next;
        box_anchor_xy_.clear();
        box_occ_.clear();
        box_next_.clear();
        for (const auto& table : box_tables_) {
            box_anchor_xy_.push_back(table.xy);
            box_occ_.push_back(table.occ);
            box_next_.push_back(table.next);
        }
    }

    void enumerate_box_configs() {
        const int num_boxes = static_cast<int>(spec_.boxes.size());
        std::vector<int> order(num_boxes);
        for (int i = 0; i < num_boxes; ++i) order[i] = i;
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            if (box_occ_[a].size() != box_occ_[b].size()) return box_occ_[a].size() < box_occ_[b].size();
            if (spec_.boxes[a].cells.size() != spec_.boxes[b].cells.size()) return spec_.boxes[a].cells.size() > spec_.boxes[b].cells.size();
            return spec_.boxes[a].name < spec_.boxes[b].name;
        });
        std::vector<int> anchors(num_boxes, -1);
        Bits empty = make_bits(area_);

        std::function<void(int, const Bits&)> dfs = [&](int depth, const Bits& occ_union) {
            if (depth == num_boxes) {
                const int cfg_id = static_cast<int>(box_cfg_anchors_.size());
                box_cfg_id_by_anchors_[anchors] = cfg_id;
                box_cfg_anchors_.push_back(anchors);
                box_cfg_union_.push_back(occ_union);
                return;
            }
            const int box_i = order[depth];
            for (int anchor_idx = 0; anchor_idx < static_cast<int>(box_occ_[box_i].size()); ++anchor_idx) {
                if (intersects(box_occ_[box_i][anchor_idx], occ_union)) continue;
                anchors[box_i] = anchor_idx;
                Bits next_union = occ_union;
                or_into(next_union, box_occ_[box_i][anchor_idx]);
                dfs(depth + 1, next_union);
                anchors[box_i] = -1;
            }
        };
        dfs(0, empty);
        if (box_cfg_anchors_.empty()) throw std::invalid_argument("no collision-free box placements exist");
    }

    void build_macrostates() {
        const int num_player_anchors = static_cast<int>(player_occ_.size());
        cfg_player_owner_.reserve(box_cfg_anchors_.size());
        for (std::size_t cfg_id = 0; cfg_id < box_cfg_anchors_.size(); ++cfg_id) {
            const Bits& union_bits = box_cfg_union_[cfg_id];
            std::vector<int> owner(num_player_anchors, -1);
            for (int start = 0; start < num_player_anchors; ++start) {
                if (owner[start] != -1 || intersects(player_occ_[start], union_bits)) continue;
                const int macro_id = static_cast<int>(macro_box_cfg_.size());
                macro_box_cfg_.push_back(static_cast<int>(cfg_id));
                std::deque<int> dq;
                dq.push_back(start);
                owner[start] = macro_id;
                while (!dq.empty()) {
                    const int p = dq.front();
                    dq.pop_front();
                    macro_anchors_flat_.push_back(p);
                    for (int nxt : player_next_[p]) {
                        if (nxt == -1 || owner[nxt] != -1) continue;
                        if (intersects(player_occ_[nxt], union_bits)) continue;
                        owner[nxt] = macro_id;
                        dq.push_back(nxt);
                    }
                }
                macro_anchor_offsets_.push_back(static_cast<int>(macro_anchors_flat_.size()));
            }
            cfg_player_owner_.push_back(std::move(owner));
        }
        if (macro_box_cfg_.empty()) throw std::invalid_argument("no valid states exist");
    }

    [[nodiscard]] std::vector<std::pair<int, std::uint32_t>> iter_macro_out_edges(int src_macro) const {
        const int cfg_id = macro_box_cfg_[src_macro];
        const auto& anchors = box_cfg_anchors_[cfg_id];
        const Bits& union_bits = box_cfg_union_[cfg_id];
        std::unordered_map<int, std::vector<std::uint32_t>> per_dest;

        for (int idx = macro_anchor_offsets_[src_macro]; idx < macro_anchor_offsets_[src_macro + 1]; ++idx) {
            const int player_anchor = macro_anchors_flat_[idx];
            for (int dir = 0; dir < 4; ++dir) {
                const int next_player = player_next_[player_anchor][dir];
                if (next_player == -1) continue;
                if (!intersects(player_occ_[next_player], union_bits)) continue;

                std::uint32_t moved_mask = 0;
                std::vector<int> frontier;
                for (int b = 0; b < static_cast<int>(spec_.boxes.size()); ++b) {
                    if (intersects(player_occ_[next_player], box_occ_[b][anchors[b]])) {
                        moved_mask |= (1u << b);
                        frontier.push_back(b);
                    }
                }
                if (frontier.empty()) continue;

                bool blocked = false;
                for (std::size_t q = 0; q < frontier.size(); ++q) {
                    const int i = frontier[q];
                    const int next_anchor = box_next_[i][anchors[i]][dir];
                    if (next_anchor == -1) {
                        blocked = true;
                        break;
                    }
                    const Bits& shifted_bits = box_occ_[i][next_anchor];
                    for (int j = 0; j < static_cast<int>(spec_.boxes.size()); ++j) {
                        if ((moved_mask >> j) & 1u) continue;
                        if (intersects(shifted_bits, box_occ_[j][anchors[j]])) {
                            moved_mask |= (1u << j);
                            frontier.push_back(j);
                        }
                    }
                }
                if (blocked) continue;

                std::vector<int> new_anchors = anchors;
                for (int i : frontier) new_anchors[i] = box_next_[i][anchors[i]][dir];
                const int dest_cfg = box_cfg_id_by_anchors_.at(new_anchors);
                const int dest_macro = cfg_player_owner_[dest_cfg][next_player];
                if (dest_macro < 0) continue;

                auto& masks = per_dest[dest_macro];
                bool keep = true;
                for (std::size_t i = 0; i < masks.size();) {
                    const std::uint32_t old_mask = masks[i];
                    if ((old_mask | moved_mask) == old_mask) {
                        keep = false;
                        break;
                    }
                    if ((old_mask | moved_mask) == moved_mask) {
                        masks.erase(masks.begin() + static_cast<long>(i));
                    } else {
                        ++i;
                    }
                }
                if (keep) masks.push_back(moved_mask);
            }
        }

        std::vector<std::pair<int, std::uint32_t>> out;
        for (const auto& [dest, masks] : per_dest) {
            for (std::uint32_t mask : masks) out.emplace_back(dest, mask);
        }
        return out;
    }

    void build_graph() {
        reverse_adj_.assign(macro_box_cfg_.size(), {});
        for (int src = 0; src < static_cast<int>(macro_box_cfg_.size()); ++src) {
            for (const auto& [dest, mask] : iter_macro_out_edges(src)) {
                reverse_adj_[dest].emplace_back(src, mask);
            }
        }
    }

    [[nodiscard]] std::vector<int> signature_from_cfg_anchors(const std::vector<int>& anchors) const {
        std::vector<int> sig;
        sig.reserve(target_box_indices_.size());
        for (int bi : target_box_indices_) sig.push_back(anchors[bi]);
        return sig;
    }

    void build_goal_buckets() {
        macro_signature_.resize(macro_box_cfg_.size());
        for (int macro = 0; macro < static_cast<int>(macro_box_cfg_.size()); ++macro) {
            const int cfg_id = macro_box_cfg_[macro];
            const auto sig = signature_from_cfg_anchors(box_cfg_anchors_[cfg_id]);
            macro_signature_[macro] = sig;
            goal_bucket_by_signature_[sig].push_back(macro);
        }
    }

    [[nodiscard]] std::vector<int> reverse_zero_one_bfs(const std::vector<int>& goals, int box_index) const {
        const std::uint32_t bit = (1u << box_index);
        std::vector<int> dist(macro_box_cfg_.size(), INF);
        std::deque<int> dq;
        for (int g : goals) {
            if (dist[g] == 0) continue;
            dist[g] = 0;
            dq.push_back(g);
        }
        while (!dq.empty()) {
            const int u = dq.front();
            dq.pop_front();
            const int du = dist[u];
            for (const auto& [pred, mask] : reverse_adj_[u]) {
                const int w = (mask & bit) ? 1 : 0;
                const int nd = du + w;
                if (nd < dist[pred]) {
                    dist[pred] = nd;
                    if (w == 0) dq.push_front(pred);
                    else dq.push_back(pred);
                }
            }
        }
        return dist;
    }

    // Cached views used throughout the implementation.
    std::vector<std::pair<int, int>> player_anchor_xy_;
    std::vector<Bits> player_occ_;
    std::vector<std::array<int, 4>> player_next_;
    std::vector<std::vector<std::pair<int, int>>> box_anchor_xy_;
    std::vector<std::vector<Bits>> box_occ_;
    std::vector<std::vector<std::array<int, 4>>> box_next_;
};

}  // namespace pushworld

namespace {

struct JsonValue {
    enum class Kind { kNull, kBoolean, kNumber, kString, kArray, kObject };

    Kind kind = Kind::kNull;
    bool bool_value = false;
    std::string string_value;
    std::vector<JsonValue> array_value;
    std::unordered_map<std::string, JsonValue> object_value;

    [[nodiscard]] bool contains(std::string_view key) const {
        if (kind != Kind::kObject) return false;
        return object_value.find(std::string(key)) != object_value.end();
    }

    [[nodiscard]] const JsonValue& require(std::string_view key) const {
        if (kind != Kind::kObject) throw std::invalid_argument("expected JSON object");
        auto it = object_value.find(std::string(key));
        if (it == object_value.end()) throw std::invalid_argument("missing JSON field: " + std::string(key));
        return it->second;
    }
};

class JsonParser {
public:
    explicit JsonParser(std::string text) : text_(std::move(text)) {}

    [[nodiscard]] JsonValue parse() {
        JsonValue value = parse_value();
        skip_ws();
        if (pos_ != text_.size()) throw std::invalid_argument("trailing JSON content");
        return value;
    }

private:
    std::string text_;
    std::size_t pos_ = 0;

    void skip_ws() {
        while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) ++pos_;
    }

    [[nodiscard]] char peek() const {
        if (pos_ >= text_.size()) throw std::invalid_argument("unexpected end of JSON");
        return text_[pos_];
    }

    [[nodiscard]] char get() {
        const char ch = peek();
        ++pos_;
        return ch;
    }

    void expect(char expected) {
        const char actual = get();
        if (actual != expected) {
            throw std::invalid_argument("expected '" + std::string(1, expected) + "' in JSON");
        }
    }

    [[nodiscard]] JsonValue parse_value() {
        skip_ws();
        const char ch = peek();
        if (ch == '{') return parse_object();
        if (ch == '[') return parse_array();
        if (ch == '"') {
            JsonValue value;
            value.kind = JsonValue::Kind::kString;
            value.string_value = parse_string();
            return value;
        }
        if (ch == 't') return parse_true();
        if (ch == 'f') return parse_false();
        if (ch == 'n') return parse_null();
        if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch))) return parse_number();
        throw std::invalid_argument("unsupported JSON token");
    }

    [[nodiscard]] JsonValue parse_object() {
        expect('{');
        JsonValue value;
        value.kind = JsonValue::Kind::kObject;
        skip_ws();
        if (peek() == '}') {
            ++pos_;
            return value;
        }
        while (true) {
            skip_ws();
            const std::string key = parse_string();
            skip_ws();
            expect(':');
            JsonValue item = parse_value();
            value.object_value.emplace(key, std::move(item));
            skip_ws();
            const char ch = get();
            if (ch == '}') break;
            if (ch != ',') throw std::invalid_argument("expected ',' or '}' in JSON object");
        }
        return value;
    }

    [[nodiscard]] JsonValue parse_array() {
        expect('[');
        JsonValue value;
        value.kind = JsonValue::Kind::kArray;
        skip_ws();
        if (peek() == ']') {
            ++pos_;
            return value;
        }
        while (true) {
            value.array_value.push_back(parse_value());
            skip_ws();
            const char ch = get();
            if (ch == ']') break;
            if (ch != ',') throw std::invalid_argument("expected ',' or ']' in JSON array");
        }
        return value;
    }

    [[nodiscard]] std::string parse_string() {
        expect('"');
        std::string out;
        while (true) {
            const char ch = get();
            if (ch == '"') break;
            if (ch == '\\') {
                const char esc = get();
                switch (esc) {
                    case '"':
                    case '\\':
                    case '/':
                        out.push_back(esc);
                        break;
                    case 'b':
                        out.push_back('\b');
                        break;
                    case 'f':
                        out.push_back('\f');
                        break;
                    case 'n':
                        out.push_back('\n');
                        break;
                    case 'r':
                        out.push_back('\r');
                        break;
                    case 't':
                        out.push_back('\t');
                        break;
                    default:
                        throw std::invalid_argument("unsupported JSON escape sequence");
                }
            } else {
                out.push_back(ch);
            }
        }
        return out;
    }

    [[nodiscard]] JsonValue parse_number() {
        const std::size_t start = pos_;
        if (text_[pos_] == '-') ++pos_;
        if (pos_ >= text_.size() || !std::isdigit(static_cast<unsigned char>(text_[pos_]))) {
            throw std::invalid_argument("invalid JSON number");
        }
        if (text_[pos_] == '0') {
            ++pos_;
        } else {
            while (pos_ < text_.size() && std::isdigit(static_cast<unsigned char>(text_[pos_]))) ++pos_;
        }
        if (pos_ < text_.size() && (text_[pos_] == '.' || text_[pos_] == 'e' || text_[pos_] == 'E')) {
            throw std::invalid_argument("floating-point JSON numbers are not supported");
        }
        JsonValue value;
        value.kind = JsonValue::Kind::kNumber;
        value.string_value = text_.substr(start, pos_ - start);
        return value;
    }

    [[nodiscard]] JsonValue parse_true() {
        if (text_.compare(pos_, 4, "true") != 0) throw std::invalid_argument("invalid JSON literal");
        pos_ += 4;
        JsonValue value;
        value.kind = JsonValue::Kind::kBoolean;
        value.bool_value = true;
        return value;
    }

    [[nodiscard]] JsonValue parse_false() {
        if (text_.compare(pos_, 5, "false") != 0) throw std::invalid_argument("invalid JSON literal");
        pos_ += 5;
        JsonValue value;
        value.kind = JsonValue::Kind::kBoolean;
        value.bool_value = false;
        return value;
    }

    [[nodiscard]] JsonValue parse_null() {
        if (text_.compare(pos_, 4, "null") != 0) throw std::invalid_argument("invalid JSON literal");
        pos_ += 4;
        JsonValue value;
        value.kind = JsonValue::Kind::kNull;
        return value;
    }
};

[[nodiscard]] long long parse_int64(const JsonValue& value) {
    if (value.kind != JsonValue::Kind::kNumber) throw std::invalid_argument("expected integer JSON number");
    std::size_t idx = 0;
    long long parsed = 0;
    try {
        parsed = std::stoll(value.string_value, &idx, 10);
    } catch (const std::exception&) {
        throw std::invalid_argument("invalid integer JSON number: " + value.string_value);
    }
    if (idx != value.string_value.size()) {
        throw std::invalid_argument("invalid integer JSON number: " + value.string_value);
    }
    return parsed;
}

[[nodiscard]] int parse_int(const JsonValue& value, const std::string& field_name) {
    const long long parsed = parse_int64(value);
    if (parsed < std::numeric_limits<int>::min() || parsed > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(field_name + " is out of range for int");
    }
    return static_cast<int>(parsed);
}

[[nodiscard]] std::uint64_t parse_u64(const JsonValue& value, const std::string& field_name) {
    if (value.kind != JsonValue::Kind::kNumber) throw std::invalid_argument("expected integer JSON number");
    std::size_t idx = 0;
    unsigned long long parsed = 0;
    try {
        parsed = std::stoull(value.string_value, &idx, 10);
    } catch (const std::exception&) {
        throw std::invalid_argument(field_name + " must be a non-negative integer");
    }
    if (idx != value.string_value.size()) {
        throw std::invalid_argument(field_name + " must be a non-negative integer");
    }
    return static_cast<std::uint64_t>(parsed);
}

[[nodiscard]] bool parse_bool(const JsonValue& value, bool default_value = false) {
    if (value.kind == JsonValue::Kind::kNull) return default_value;
    if (value.kind != JsonValue::Kind::kBoolean) throw std::invalid_argument("expected JSON boolean");
    return value.bool_value;
}

[[nodiscard]] std::string parse_string_field(const JsonValue& value, const std::string& field_name) {
    if (value.kind != JsonValue::Kind::kString) throw std::invalid_argument(field_name + " must be a JSON string");
    return value.string_value;
}

[[nodiscard]] std::vector<std::pair<int, int>> parse_cells(const JsonValue& value, const std::string& field_name) {
    if (value.kind != JsonValue::Kind::kArray) throw std::invalid_argument(field_name + " must be a JSON array");
    std::vector<std::pair<int, int>> cells;
    cells.reserve(value.array_value.size());
    for (const JsonValue& cell : value.array_value) {
        if (cell.kind != JsonValue::Kind::kArray || cell.array_value.size() != 2) {
            throw std::invalid_argument(field_name + " entries must be [x, y] arrays");
        }
        cells.emplace_back(parse_int(cell.array_value[0], field_name), parse_int(cell.array_value[1], field_name));
    }
    return cells;
}

[[nodiscard]] pushworld::Bits decimal_to_bits(const std::string& text, int area) {
    if (text.empty()) throw std::invalid_argument("wall mask cannot be empty");
    if (text[0] == '-') throw std::invalid_argument("wall mask must be non-negative");
    for (char ch : text) {
        if (!std::isdigit(static_cast<unsigned char>(ch))) {
            throw std::invalid_argument("wall mask must be a base-10 integer");
        }
    }

    std::string digits = text;
    const std::size_t first_non_zero = digits.find_first_not_of('0');
    if (first_non_zero == std::string::npos) return pushworld::make_bits(area);
    digits.erase(0, first_non_zero);

    pushworld::Bits bits = pushworld::make_bits(area);
    int bit_index = 0;
    while (!(digits.size() == 1 && digits[0] == '0')) {
        int remainder = 0;
        std::string quotient;
        quotient.reserve(digits.size());
        for (char ch : digits) {
            const int cur = remainder * 10 + (ch - '0');
            const int q = cur / 2;
            remainder = cur % 2;
            if (!quotient.empty() || q != 0) quotient.push_back(static_cast<char>('0' + q));
        }
        if (remainder != 0) {
            if (bit_index >= area) throw std::invalid_argument("wall mask exceeds board area");
            pushworld::set_bit(bits, bit_index);
        }
        ++bit_index;
        digits = quotient.empty() ? "0" : quotient;
    }
    return bits;
}

[[nodiscard]] bool bit_test(const pushworld::Bits& bits, int index) {
    return ((bits[static_cast<std::size_t>(index / 64)] >> (index % 64)) & 1ull) != 0ull;
}

[[nodiscard]] std::string bits_to_decimal_string(const pushworld::Bits& bits, int area) {
    std::vector<int> digits(1, 0);
    for (int index = area - 1; index >= 0; --index) {
        int carry = bit_test(bits, index) ? 1 : 0;
        for (std::size_t i = 0; i < digits.size(); ++i) {
            const int value = digits[i] * 2 + carry;
            digits[i] = value % 10;
            carry = value / 10;
        }
        while (carry > 0) {
            digits.push_back(carry % 10);
            carry /= 10;
        }
    }
    std::string out;
    out.reserve(digits.size());
    for (auto it = digits.rbegin(); it != digits.rend(); ++it) out.push_back(static_cast<char>('0' + *it));
    return out;
}

[[nodiscard]] std::string json_escape(const std::string& input) {
    std::ostringstream oss;
    for (unsigned char ch : input) {
        switch (ch) {
            case '"':
                oss << "\\\"";
                break;
            case '\\':
                oss << "\\\\";
                break;
            case '\b':
                oss << "\\b";
                break;
            case '\f':
                oss << "\\f";
                break;
            case '\n':
                oss << "\\n";
                break;
            case '\r':
                oss << "\\r";
                break;
            case '\t':
                oss << "\\t";
                break;
            default:
                if (ch < 0x20) {
                    oss << "\\u00" << std::hex << std::uppercase << std::setw(2) << std::setfill('0')
                        << static_cast<int>(ch) << std::dec << std::nouppercase << std::setfill(' ');
                } else {
                    oss << static_cast<char>(ch);
                }
        }
    }
    return oss.str();
}

[[nodiscard]] pushworld::Polyomino parse_box(const JsonValue& box_value, std::size_t box_index) {
    if (box_value.kind != JsonValue::Kind::kObject) throw std::invalid_argument("box entries must be JSON objects");
    std::string default_name(1, static_cast<char>('a' + static_cast<int>(box_index)));
    std::string name = default_name;
    if (box_value.contains("name")) name = parse_string_field(box_value.require("name"), "box.name");
    return pushworld::Polyomino{
        name,
        parse_cells(box_value.require("cells"), "boxes[].cells"),
        box_value.contains("has_target") ? parse_bool(box_value.require("has_target")) : false,
    };
}

[[nodiscard]] pushworld::BoardSpec load_spec_from_json_file(
    const std::string& path,
    const std::optional<std::string>& override_walls_text
) {
    std::ifstream in(path);
    if (!in) throw std::invalid_argument("could not open spec file: " + path);
    std::ostringstream buffer;
    buffer << in.rdbuf();

    JsonParser parser(buffer.str());
    const JsonValue root = parser.parse();
    if (root.kind != JsonValue::Kind::kObject) throw std::invalid_argument("spec must be a JSON object");

    pushworld::BoardSpec spec;
    spec.width = parse_int(root.require("width"), "width");
    spec.height = parse_int(root.require("height"), "height");
    spec.profile_scope = root.contains("profile_scope")
        ? parse_string_field(root.require("profile_scope"), "profile_scope")
        : "all";
    if (root.contains("wall_seed") && root.require("wall_seed").kind != JsonValue::Kind::kNull) {
        spec.wall_seed = parse_u64(root.require("wall_seed"), "wall_seed");
    }
    spec.player = pushworld::Polyomino{
        "1",
        parse_cells(root.require("player_shape"), "player_shape"),
        false,
    };

    const JsonValue& boxes_value = root.require("boxes");
    if (boxes_value.kind != JsonValue::Kind::kArray) throw std::invalid_argument("boxes must be a JSON array");
    spec.boxes.reserve(boxes_value.array_value.size());
    for (std::size_t i = 0; i < boxes_value.array_value.size(); ++i) {
        spec.boxes.push_back(parse_box(boxes_value.array_value[i], i));
    }

    const int area = spec.width * spec.height;
    if (override_walls_text.has_value()) {
        spec.walls = decimal_to_bits(*override_walls_text, area);
    } else if (root.contains("walls")) {
        const JsonValue& walls_value = root.require("walls");
        if (walls_value.kind != JsonValue::Kind::kNumber) throw std::invalid_argument("walls must be a JSON integer");
        spec.walls = decimal_to_bits(walls_value.string_value, area);
    } else {
        const int wall_count = parse_int(root.require("wall_count"), "wall_count");
        if (!spec.wall_seed.has_value()) throw std::invalid_argument("wall_seed is required with wall_count");
        spec.walls = pushworld::generate_walls(spec.width, spec.height, wall_count, *spec.wall_seed);
    }

    return spec;
}

[[nodiscard]] std::string hardest_result_to_json(const pushworld::HardestPuzzleResult& result) {
    std::ostringstream oss;
    oss << "{";

    oss << "\"profile\":[";
    for (std::size_t i = 0; i < result.profile.size(); ++i) {
        if (i) oss << ",";
        oss << result.profile[i];
    }
    oss << "]";

    oss << ",\"per_box_min_pushes\":{";
    for (std::size_t i = 0; i < result.board.boxes.size(); ++i) {
        if (i) oss << ",";
        oss << "\"" << json_escape(result.board.boxes[i].name) << "\":" << result.per_box_min_pushes[i];
    }
    oss << "}";

    oss << ",\"start_state\":{";
    oss << "\"player_anchor\":[" << result.start_state.player_anchor.first << "," << result.start_state.player_anchor.second << "]";
    oss << ",\"box_anchors\":[";
    for (std::size_t i = 0; i < result.start_state.box_anchors.size(); ++i) {
        if (i) oss << ",";
        const auto [x, y] = result.start_state.box_anchors[i];
        oss << "[" << x << "," << y << "]";
    }
    oss << "]}";

    oss << ",\"target_anchors\":{";
    bool first_target = true;
    for (std::size_t i = 0; i < result.board.boxes.size(); ++i) {
        if (!result.target_anchors_by_box[i].has_value()) continue;
        if (!first_target) oss << ",";
        first_target = false;
        const auto [x, y] = *result.target_anchors_by_box[i];
        oss << "\"" << json_escape(result.board.boxes[i].name) << "\":[" << x << "," << y << "]";
    }
    oss << "}";

    const int area = result.board.width * result.board.height;
    oss << ",\"width\":" << result.board.width;
    oss << ",\"height\":" << result.board.height;
    oss << ",\"walls\":" << bits_to_decimal_string(result.board.walls, area);
    oss << ",\"wall_count\":" << pushworld::popcount(result.board.walls);
    oss << ",\"wall_seed\":";
    if (result.board.wall_seed.has_value()) {
        oss << *result.board.wall_seed;
    } else {
        oss << "null";
    }
    oss << ",\"rendered\":\"" << json_escape(result.render()) << "\"";
    oss << "}";

    return oss.str();
}

[[nodiscard]] std::string usage_text() {
    return
        "Usage:\n"
        "  pushworld_solver_cpp hardest <spec.json> [wall_mask]\n"
        "  pushworld_solver_cpp <spec.json> [wall_mask]\n";
}

int run_hardest_cli(const std::string& spec_path, const std::optional<std::string>& override_walls_text) {
    const pushworld::BoardSpec spec = load_spec_from_json_file(spec_path, override_walls_text);
    pushworld::PushWorldEngine engine(spec);
    const auto result = engine.find_hardest_puzzle();
    if (!result.has_value()) {
        std::cerr << "No solvable puzzle found.\n";
        return 1;
    }
    std::cout << hardest_result_to_json(*result) << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc <= 1) {
            std::cerr << usage_text();
            return 1;
        }

        const std::string_view first = argv[1];
        if (first == "-h" || first == "--help") {
            std::cout << usage_text();
            return 0;
        }

        if (first == "hardest") {
            if (argc < 3 || argc > 4) throw std::invalid_argument("invalid arguments for hardest");
            return run_hardest_cli(argv[2], argc == 4 ? std::optional<std::string>(argv[3]) : std::nullopt);
        }

        if (argc == 2 || argc == 3) {
            return run_hardest_cli(argv[1], argc == 3 ? std::optional<std::string>(argv[2]) : std::nullopt);
        }

        throw std::invalid_argument("unknown command");
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << '\n';
        return 2;
    }
}
