// Build inside the google-deepmind/pushworld repo with something like:
// g++ -std=c++20 -O3 -Icpp/include advanced_pushworld_recursive_solver.cpp \
//     cpp/src/pushworld_puzzle.cc \
//     cpp/src/heuristics/domain_transition_graph.cc \
//     cpp/src/heuristics/recursive_graph_distance.cc \
//     -o advanced_pushworld_recursive_solver
//
// Usage:
//   ./advanced_pushworld_recursive_solver macro-recursive benchmark/puzzles/level1/2\ Obstacle.pwp
//
// The program prints either a UDLR plan or NO SOLUTION.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pushworld_puzzle.h"
#include "heuristics/domain_transition_graph.h"
#include "heuristics/recursive_graph_distance.h"
#include "pushworld_recursive_subset_probes.hpp"

using namespace pushworld;

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr std::size_t kReducedExpansionLimit = 3000;
constexpr double kReducedTimeLimitSec = 0.02;
constexpr std::size_t kRecursiveEstimateCacheLimit = 2048;
constexpr double kNoveltyWeight = 1000000.0;

double NoveltyWeightedScore(const float novelty,
                           const float base_score,
                           const double novelty_weight) {
  return static_cast<double>(novelty) * novelty_weight +
         static_cast<double>(base_score);
}

struct SearchStats {
  size_t expansions = 0;
  size_t generated = 0;
  size_t reopened = 0;
};

struct Node {
  State state;
  float g = 0.0f;
  float h = 0.0f;
  int parent = -1;
  std::string segment;  // primitive action string from parent to here.
};

struct OpenItem {
  double priority = 0.0;
  float f = 0.0f;
  float h = 0.0f;
  float novelty = 3.0f;
  float g = 0.0f;
  int node_idx = -1;

  bool operator<(const OpenItem& other) const {
    if (priority != other.priority) return priority > other.priority;
    if (f != other.f) return f > other.f;
    if (h != other.h) return h > other.h;
    return g < other.g;  // prefer deeper nodes on ties.
  }
};

struct PositionPairHash {
  std::size_t operator()(const std::pair<Position2D, Position2D>& pair) const {
    std::size_t seed = 0;
    boost::hash_combine(seed, pair.first);
    boost::hash_combine(seed, pair.second);
    return seed;
  }
};

class NoveltyTracker {
 public:
  explicit NoveltyTracker(int state_size)
      : visited_positions_(state_size),
        visited_position_pairs_(state_size,
                                std::vector<std::unordered_set<std::pair<Position2D, Position2D>, PositionPairHash>>(state_size)) {}

  float Observe(const RelativeState& relative_state) {
    float novelty = 3.0f;
    for (int i : relative_state.moved_object_indices) {
      const auto p_i = relative_state.state[i];
      if (visited_positions_[i].insert(p_i).second) {
        novelty = 1.0f;
      }
      for (int j = 0; j < i; ++j) {
        const auto p_j = relative_state.state[j];
        if (visited_position_pairs_[j][i].insert({p_j, p_i}).second) {
          novelty = std::min(novelty, 2.0f);
        }
      }
      for (int j = i + 1; j < static_cast<int>(visited_positions_.size()); ++j) {
        const auto p_j = relative_state.state[j];
        if (visited_position_pairs_[i][j].insert({p_i, p_j}).second) {
          novelty = std::min(novelty, 2.0f);
        }
      }
    }
    return novelty;
  }

 private:
  std::vector<std::unordered_set<Position2D>> visited_positions_;
  std::vector<std::vector<std::unordered_set<std::pair<Position2D, Position2D>, PositionPairHash>>> visited_position_pairs_;
};

struct Successor {
  RelativeState relative_state;
  float cost = 1.0f;
  std::string segment;
};

struct HeuristicResult {
  float h = kInf;
  bool dead = false;
};

class StaticDistanceHeuristic {
 public:
  explicit StaticDistanceHeuristic(const std::shared_ptr<PushWorldPuzzle>& puzzle)
      : puzzle_(puzzle), graphs_(heuristic::build_feasible_movement_graphs(*puzzle)) {
    for (const auto& [object_id, graph] : graphs_) {
      path_distances_.emplace(object_id, heuristic::PathDistances(graph));
    }
    BuildAgentComponents();
  }

  float Estimate(const State& state) const {
    float h = 0.0f;
    const Goal& goal = puzzle_->getGoal();
    for (int object_id = 1; object_id <= static_cast<int>(goal.size()); ++object_id) {
      if (state[object_id] == goal[object_id - 1]) continue;
      float d = path_distances_.at(object_id).getDistance(state[object_id], goal[object_id - 1]);
      if (!std::isfinite(d)) return kInf;
      h += d;
    }
    return h;
  }

  bool ProvablyDead(const State& state) const {
    const Goal& goal = puzzle_->getGoal();
    for (int object_id = 1; object_id <= static_cast<int>(goal.size()); ++object_id) {
      Position2D cur = state[object_id];
      Position2D target = goal[object_id - 1];
      if (cur == target) continue;
      const auto graph_it = graphs_.find(object_id);
      if (graph_it == graphs_.end()) return true;
      const auto& graph = *graph_it->second;
      const auto node_it = graph.find(cur);
      if (node_it == graph.end()) return true;
      if (path_distances_.at(object_id).getDistance(cur, target) == kInf) return true;
      if (node_it->second.empty()) return true;
    }
    return false;
  }

  float GatePenalty(const State& state) const {
    float penalty = 0.0f;
    const Goal& goal = puzzle_->getGoal();
    Position2D agent_pos = state[AGENT];
    auto comp_it = agent_component_.find(agent_pos);
    if (comp_it == agent_component_.end()) return 0.0f;
    int agent_comp = comp_it->second;

    const auto& collisions = puzzle_->getObjectCollisions();
    const auto& agent_graph = *graphs_.at(AGENT);

    for (int object_id = 1; object_id <= static_cast<int>(goal.size()); ++object_id) {
      Position2D cur = state[object_id];
      Position2D target = goal[object_id - 1];
      if (cur == target) continue;
      const auto& object_graph = *graphs_.at(object_id);
      const auto node_it = object_graph.find(cur);
      if (node_it == object_graph.end()) continue;

      float best_remaining = kInf;
      std::vector<Position2D> best_nexts;
      for (Position2D nxt : node_it->second) {
        float rem = path_distances_.at(object_id).getDistance(nxt, target);
        if (rem < best_remaining) {
          best_remaining = rem;
          best_nexts.clear();
          best_nexts.push_back(nxt);
        } else if (rem == best_remaining) {
          best_nexts.push_back(nxt);
        }
      }
      if (best_nexts.empty()) continue;

      bool directly_accessible = false;
      for (Position2D nxt : best_nexts) {
        Position2D disp = nxt - cur;
        Action action = DISPLACEMENTS_TO_ACTIONS.at(disp);
        const auto& rel_positions = collisions.dynamic_collisions[action][AGENT][object_id];
        for (Position2D rel : rel_positions) {
          Position2D contact = cur + rel;
          Position2D contact_after = contact + disp;
          auto comp_contact = agent_component_.find(contact);
          if (comp_contact == agent_component_.end() || comp_contact->second != agent_comp) continue;
          auto agent_node = agent_graph.find(contact);
          if (agent_node == agent_graph.end()) continue;
          if (agent_node->second.find(contact_after) != agent_node->second.end()) {
            directly_accessible = true;
            break;
          }
        }
        if (directly_accessible) break;
      }
      if (!directly_accessible) penalty += 1.0f;
    }
    return penalty;
  }

  float ImmediateBlockerPenalty(const State& state) const {
    float penalty = 0.0f;
    const Goal& goal = puzzle_->getGoal();
    std::unordered_set<Position2D> occupied(state.begin() + 1, state.end());
    for (int object_id = 1; object_id <= static_cast<int>(goal.size()); ++object_id) {
      Position2D cur = state[object_id];
      Position2D target = goal[object_id - 1];
      if (cur == target) continue;
      const auto& object_graph = *graphs_.at(object_id);
      auto node_it = object_graph.find(cur);
      if (node_it == object_graph.end() || node_it->second.empty()) continue;
      float best_remaining = kInf;
      bool any_free_best = false;
      for (Position2D nxt : node_it->second) {
        float rem = path_distances_.at(object_id).getDistance(nxt, target);
        if (rem < best_remaining) {
          best_remaining = rem;
          any_free_best = (occupied.find(nxt) == occupied.end());
        } else if (rem == best_remaining) {
          any_free_best = any_free_best || (occupied.find(nxt) == occupied.end());
        }
      }
      if (!any_free_best) penalty += 1.0f;
    }
    return penalty;
  }

  const std::unordered_map<int, std::shared_ptr<heuristic::FeasibleMovementGraph>>& graphs() const {
    return graphs_;
  }

  const std::unordered_map<int, heuristic::PathDistances>& path_distances() const {
    return path_distances_;
  }

 private:
  void BuildAgentComponents() {
    const auto& graph = *graphs_.at(AGENT);
    int comp = 0;
    std::deque<Position2D> dq;
    for (const auto& [start, _] : graph) {
      if (agent_component_.find(start) != agent_component_.end()) continue;
      agent_component_[start] = comp;
      dq.push_back(start);
      while (!dq.empty()) {
        Position2D u = dq.front();
        dq.pop_front();
        auto it = graph.find(u);
        if (it == graph.end()) continue;
        for (Position2D v : it->second) {
          if (agent_component_.emplace(v, comp).second) {
            dq.push_back(v);
          }
        }
        // Because graph is directed, also traverse reverse neighbors by scan.
        for (const auto& [x, nbrs] : graph) {
          if (nbrs.find(u) != nbrs.end()) {
            if (agent_component_.emplace(x, comp).second) {
              dq.push_back(x);
            }
          }
        }
      }
      ++comp;
    }
  }

  std::shared_ptr<PushWorldPuzzle> puzzle_;
  std::unordered_map<int, std::shared_ptr<heuristic::FeasibleMovementGraph>> graphs_;
  std::unordered_map<int, heuristic::PathDistances> path_distances_;
  std::unordered_map<Position2D, int> agent_component_;
};

std::optional<std::string> SolveReducedPuzzleQuickly(
    const std::shared_ptr<PushWorldPuzzle>& puzzle);

class PortfolioHeuristic {
 public:
  PortfolioHeuristic(const std::shared_ptr<PushWorldPuzzle>& puzzle,
                     bool rgd_fewest_tools,
                     bool use_recursive_subset = false,
                     float recursive_scale = 1.0f)
      : puzzle_(puzzle),
        static_h_(puzzle),
        rgd_(puzzle, rgd_fewest_tools),
        use_recursive_subset_(use_recursive_subset),
        recursive_scale_(recursive_scale),
        reduced_solver_(SolveReducedPuzzleQuickly) {}

  HeuristicResult Evaluate(const RelativeState& relative_state, bool use_gate_penalty = true) {
    HeuristicResult out;
    if (static_h_.ProvablyDead(relative_state.state)) {
      out.dead = true;
      out.h = kInf;
      return out;
    }

    const float h_static = static_h_.Estimate(relative_state.state);
    if (!std::isfinite(h_static)) {
      out.dead = true;
      out.h = kInf;
      return out;
    }

    float h_rgd = rgd_.estimate_cost_to_goal(relative_state);
    float h_mix = h_static + static_h_.ImmediateBlockerPenalty(relative_state.state);
    if (use_gate_penalty) {
      h_mix += static_h_.GatePenalty(relative_state.state);
    }
    out.h = std::max(h_rgd, h_mix);
    if (use_recursive_subset_) {
      const float h_recursive =
          EstimateRecursiveSubset(relative_state.state);
      if (std::isfinite(h_recursive)) {
        out.h = std::max(out.h, recursive_scale_ * h_recursive);
      }
    }
    return out;
  }

 private:
  float EstimateRecursiveSubset(const State& state) {
    auto cache_it = recursive_cache_.find(state);
    if (cache_it != recursive_cache_.end()) {
      return cache_it->second;
    }

    float total = 0.0f;
    const Goal& goal = puzzle_->getGoal();
    for (int object_id = 1; object_id <= static_cast<int>(goal.size()); ++object_id) {
      if (state[object_id] == goal[object_id - 1]) continue;
      total += recursive_subgoals::RecursiveSubsetEstimate(
          *puzzle_, state, object_id, reduced_solver_,
          /*max_extra_tools=*/2,
          /*affordance_depth=*/2,
          /*candidate_cap=*/8,
          /*blocker_probe_cap=*/2,
          /*blocker_penalty=*/1.5f);
    }

    if (recursive_cache_.size() >= kRecursiveEstimateCacheLimit) {
      recursive_cache_.clear();
    }
    recursive_cache_.emplace(state, total);
    return total;
  }

  std::shared_ptr<PushWorldPuzzle> puzzle_;
  StaticDistanceHeuristic static_h_;
  heuristic::RecursiveGraphDistanceHeuristic rgd_;
  bool use_recursive_subset_ = false;
  float recursive_scale_ = 1.0f;
  recursive_subgoals::SolverCallback reduced_solver_;
  std::unordered_map<State, float, StateHash> recursive_cache_;
};

std::string ReconstructPlan(const std::vector<Node>& nodes, int goal_idx) {
  std::vector<std::string> segments;
  int idx = goal_idx;
  while (idx > 0) {
    segments.push_back(nodes[idx].segment);
    idx = nodes[idx].parent;
  }
  std::reverse(segments.begin(), segments.end());
  std::string plan;
  size_t total = 0;
  for (const auto& s : segments) total += s.size();
  plan.reserve(total);
  for (const auto& s : segments) plan += s;
  return plan;
}

std::vector<Successor> GeneratePrimitiveSuccessors(const PushWorldPuzzle& puzzle,
                                                  const State& state) {
  std::vector<Successor> out;
  out.reserve(NUM_ACTIONS);
  for (int a = 0; a < NUM_ACTIONS; ++a) {
    RelativeState next = puzzle.getNextState(state, a);
    if (next.state == state) continue;
    Successor s;
    s.relative_state = std::move(next);
    s.cost = 1.0f;
    s.segment.push_back(ACTION_TO_CHAR[a]);
    out.push_back(std::move(s));
  }
  return out;
}

std::vector<Successor> GenerateMacroSuccessors(const PushWorldPuzzle& puzzle,
                                               const State& state) {
  struct WalkNode {
    Position2D agent_pos;
    std::string path;
  };

  std::vector<Successor> successors;
  std::unordered_map<State, size_t, StateHash> best_idx;

  std::deque<WalkNode> q;
  std::unordered_set<Position2D> seen_agent_positions;
  q.push_back({state[AGENT], ""});
  seen_agent_positions.insert(state[AGENT]);

  while (!q.empty()) {
    WalkNode cur = q.front();
    q.pop_front();
    State walk_state = state;
    walk_state[AGENT] = cur.agent_pos;

    for (int a = 0; a < NUM_ACTIONS; ++a) {
      RelativeState nxt = puzzle.getNextState(walk_state, a);
      if (nxt.state == walk_state) continue;

      bool pure_walk = (nxt.moved_object_indices.size() == 1 &&
                        nxt.moved_object_indices[0] == AGENT);
      if (pure_walk) {
        Position2D new_agent = nxt.state[AGENT];
        if (seen_agent_positions.insert(new_agent).second) {
          WalkNode wn{new_agent, cur.path + ACTION_TO_CHAR[a]};
          q.push_back(std::move(wn));
        }
        continue;
      }

      Successor succ;
      succ.relative_state = std::move(nxt);
      succ.cost = static_cast<float>(cur.path.size() + 1);
      succ.segment = cur.path + ACTION_TO_CHAR[a];

      auto it = best_idx.find(succ.relative_state.state);
      if (it == best_idx.end()) {
        best_idx.emplace(succ.relative_state.state, successors.size());
        successors.push_back(std::move(succ));
      } else {
        Successor& old = successors[it->second];
        if (succ.cost < old.cost || (succ.cost == old.cost && succ.segment.size() < old.segment.size())) {
          old = std::move(succ);
        }
      }
    }
  }
  return successors;
}

std::optional<std::string> SolveReducedPuzzleQuickly(
    const std::shared_ptr<PushWorldPuzzle>& puzzle) {
  StaticDistanceHeuristic heuristic(puzzle);
  const State& initial_state = puzzle->getInitialState();
  const float root_h = heuristic.Estimate(initial_state);
  if (heuristic.ProvablyDead(initial_state) || !std::isfinite(root_h)) {
    return std::nullopt;
  }

  struct ReducedOpenItem {
    float f = 0.0f;
    float h = 0.0f;
    float g = 0.0f;
    int node_idx = -1;

    bool operator<(const ReducedOpenItem& other) const {
      if (f != other.f) return f > other.f;
      if (h != other.h) return h > other.h;
      return g < other.g;
    }
  };

  const auto start_time = std::chrono::steady_clock::now();
  std::vector<Node> nodes;
  nodes.reserve(4096);
  nodes.push_back({initial_state, 0.0f, root_h, -1, ""});

  std::priority_queue<ReducedOpenItem> open;
  std::unordered_map<State, float, StateHash> best_g;
  best_g.emplace(initial_state, 0.0f);
  open.push({root_h, root_h, 0.0f, 0});

  std::size_t expansions = 0;
  while (!open.empty() && expansions < kReducedExpansionLimit) {
    const auto now = std::chrono::steady_clock::now();
    const double elapsed_sec =
        std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time)
            .count();
    if (elapsed_sec > kReducedTimeLimitSec) {
      return std::nullopt;
    }

    ReducedOpenItem item = open.top();
    open.pop();
    const Node& cur = nodes[item.node_idx];
    auto best_it = best_g.find(cur.state);
    if (best_it == best_g.end() || best_it->second != cur.g) continue;
    if (puzzle->satisfiesGoal(cur.state)) {
      return ReconstructPlan(nodes, item.node_idx);
    }

    ++expansions;
    std::vector<Successor> succs = GenerateMacroSuccessors(*puzzle, cur.state);
    for (Successor& succ : succs) {
      const float ng = cur.g + succ.cost;
      auto old = best_g.find(succ.relative_state.state);
      if (old != best_g.end() && old->second <= ng) continue;

      const float h =
          heuristic.Estimate(succ.relative_state.state) +
          heuristic.ImmediateBlockerPenalty(succ.relative_state.state) +
          heuristic.GatePenalty(succ.relative_state.state);
      if (!std::isfinite(h)) continue;

      best_g[succ.relative_state.state] = ng;
      const int new_idx = static_cast<int>(nodes.size());
      nodes.push_back({succ.relative_state.state, ng, h, item.node_idx, succ.segment});
      open.push({ng + 1.25f * h, h, ng, new_idx});

      if (puzzle->satisfiesGoal(succ.relative_state.state)) {
        return ReconstructPlan(nodes, new_idx);
      }
    }
  }

  return std::nullopt;
}

enum class SearchStyle {
  kPrimitive,
  kMacro,
};

struct SolverConfig {
  SearchStyle style = SearchStyle::kMacro;
  bool rgd_fewest_tools = true;
  float weight = 1.6f;
  double novelty_weight = kNoveltyWeight;
  bool use_gate_penalty = true;
  bool use_recursive_subset = false;
  float recursive_scale = 1.0f;
  size_t expansion_limit = 20000000;
  double time_limit_sec = 60.0;
};

struct SolveOutcome {
  std::optional<std::string> plan;
  SearchStats stats;
  double elapsed_seconds = 0.0;
  bool timed_out = false;
  bool hit_expansion_limit = false;
  bool exhausted = false;
};

SolveOutcome Solve(const std::shared_ptr<PushWorldPuzzle>& puzzle,
                   const SolverConfig& cfg) {
  SearchStats stats;
  PortfolioHeuristic heuristic(
      puzzle,
      cfg.rgd_fewest_tools,
      cfg.use_recursive_subset,
      cfg.recursive_scale);
  NoveltyTracker novelty_tracker(static_cast<int>(puzzle->getInitialState().size()));
  const auto start_time = std::chrono::steady_clock::now();

  auto finish = [&](std::optional<std::string> plan,
                    bool timed_out,
                    bool hit_expansion_limit,
                    bool exhausted) {
    const auto end_time = std::chrono::steady_clock::now();
    SolveOutcome outcome;
    outcome.plan = std::move(plan);
    outcome.stats = stats;
    outcome.elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(end_time - start_time)
            .count();
    outcome.timed_out = timed_out;
    outcome.hit_expansion_limit = hit_expansion_limit;
    outcome.exhausted = exhausted;
    return outcome;
  };

  auto make_successors = [&](const State& s) {
    if (cfg.style == SearchStyle::kPrimitive) return GeneratePrimitiveSuccessors(*puzzle, s);
    return GenerateMacroSuccessors(*puzzle, s);
  };

  RelativeState root_rs;
  root_rs.state = puzzle->getInitialState();
  root_rs.moved_object_indices.resize(root_rs.state.size());
  for (int i = 0; i < static_cast<int>(root_rs.state.size()); ++i) {
    root_rs.moved_object_indices[i] = i;
  }
  HeuristicResult root_eval = heuristic.Evaluate(root_rs, cfg.use_gate_penalty);
  if (root_eval.dead || !std::isfinite(root_eval.h)) {
    return finish(std::nullopt, false, false, true);
  }

  std::vector<Node> nodes;
  nodes.reserve(100000);
  nodes.push_back({root_rs.state, 0.0f, root_eval.h, -1, ""});

  std::priority_queue<OpenItem> open;
  std::unordered_map<State, float, StateHash> best_g;
  best_g.emplace(root_rs.state, 0.0f);

  float root_novelty = novelty_tracker.Observe(root_rs);
  const float root_f = root_eval.h;
  open.push({NoveltyWeightedScore(root_novelty, root_f, cfg.novelty_weight),
             root_f,
             root_eval.h,
             root_novelty,
             0.0f,
             0});

  while (!open.empty()) {
    if (stats.expansions >= cfg.expansion_limit) {
      return finish(std::nullopt, false, true, false);
    }
    const auto now = std::chrono::steady_clock::now();
    const double elapsed_sec = std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time).count();
    if (elapsed_sec > cfg.time_limit_sec) {
      return finish(std::nullopt, true, false, false);
    }

    OpenItem item = open.top();
    open.pop();
    const Node& cur = nodes[item.node_idx];
    auto bg_it = best_g.find(cur.state);
    if (bg_it == best_g.end() || bg_it->second != cur.g) continue;
    if (puzzle->satisfiesGoal(cur.state)) {
      return finish(ReconstructPlan(nodes, item.node_idx), false, false, false);
    }

    ++stats.expansions;
    std::vector<Successor> succs = make_successors(cur.state);
    for (Successor& succ : succs) {
      ++stats.generated;
      float ng = cur.g + succ.cost;
      auto old = best_g.find(succ.relative_state.state);
      if (old != best_g.end() && old->second <= ng) continue;

      HeuristicResult eval = heuristic.Evaluate(succ.relative_state, cfg.use_gate_penalty);
      if (eval.dead || !std::isfinite(eval.h)) continue;

      if (old != best_g.end()) ++stats.reopened;
      best_g[succ.relative_state.state] = ng;
      float novelty = novelty_tracker.Observe(succ.relative_state);
      const float f = ng + cfg.weight * eval.h;
      int new_idx = static_cast<int>(nodes.size());
      nodes.push_back({succ.relative_state.state, ng, eval.h, item.node_idx, succ.segment});
      open.push({NoveltyWeightedScore(novelty, f, cfg.novelty_weight),
                 f,
                 eval.h,
                 novelty,
                 ng,
                 new_idx});

      if (puzzle->satisfiesGoal(succ.relative_state.state)) {
        return finish(ReconstructPlan(nodes, new_idx), false, false, false);
      }
    }
  }

  return finish(std::nullopt, false, false, true);
}

SolverConfig ConfigFromMode(const std::string& mode) {
  SolverConfig cfg;
  if (mode == "primitive-rgd") {
    cfg.style = SearchStyle::kPrimitive;
    cfg.rgd_fewest_tools = true;
    cfg.weight = 1.5f;
    cfg.use_gate_penalty = false;
  } else if (mode == "macro-rgd") {
    cfg.style = SearchStyle::kMacro;
    cfg.rgd_fewest_tools = true;
    cfg.weight = 1.5f;
    cfg.use_gate_penalty = false;
  } else if (mode == "macro-rgd-deep") {
    cfg.style = SearchStyle::kMacro;
    cfg.rgd_fewest_tools = false;
    cfg.weight = 1.35f;
    cfg.use_gate_penalty = false;
    cfg.expansion_limit = 10000000;
  } else if (mode == "macro-recursive") {
    cfg.style = SearchStyle::kMacro;
    cfg.rgd_fewest_tools = true;
    cfg.weight = 1.45f;
    cfg.use_gate_penalty = true;
    cfg.use_recursive_subset = true;
    cfg.recursive_scale = 1.0f;
  } else if (mode == "macro-recursive-safe") {
    cfg.style = SearchStyle::kMacro;
    cfg.rgd_fewest_tools = true;
    cfg.weight = 1.25f;
    cfg.use_gate_penalty = true;
    cfg.use_recursive_subset = true;
    cfg.recursive_scale = 0.9f;
  } else {
    throw std::invalid_argument(
        "Unknown mode. Expected one of: primitive-rgd, macro-rgd, macro-rgd-deep, macro-recursive, macro-recursive-safe");
  }
  return cfg;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3 && argc != 4) {
      std::cerr << "Usage: advanced_pushworld_recursive_solver <mode> <puzzle.pwp> [time_limit_seconds]\n"
                   "Modes:\n"
                   "  primitive-rgd        Primitive weighted A* with fast RGD.\n"
                   "  macro-rgd            Macro weighted A* with fast RGD.\n"
                   "  macro-rgd-deep       Macro weighted A* with full-tool RGD.\n"
                   "  macro-recursive      Recommended: novelty-weighted macro weighted A* with recursive subset probes.\n"
                   "  macro-recursive-safe Same as recursive but with lower heuristic weight.\n";
      return 1;
    }

    std::shared_ptr<PushWorldPuzzle> puzzle = std::make_shared<PushWorldPuzzle>(argv[2]);
    SolverConfig cfg = ConfigFromMode(argv[1]);
    if (argc == 4) {
      cfg.time_limit_sec = std::stod(argv[3]);
    }

    SolveOutcome outcome = Solve(puzzle, cfg);
    std::cerr << "expansions=" << outcome.stats.expansions
              << " generated=" << outcome.stats.generated
              << " reopened=" << outcome.stats.reopened
              << " elapsed_seconds=" << outcome.elapsed_seconds
              << " timed_out=" << (outcome.timed_out ? 1 : 0)
              << " hit_expansion_limit=" << (outcome.hit_expansion_limit ? 1 : 0)
              << " exhausted=" << (outcome.exhausted ? 1 : 0) << '\n';
    if (!outcome.plan.has_value()) {
      std::cout << "NO SOLUTION\n";
      return 0;
    }
    std::cout << *outcome.plan << '\n';
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << '\n';
    return 1;
  }
  return 0;
}
