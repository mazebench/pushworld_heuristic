// Build inside the google-deepmind/pushworld repo with something like:
// g++ -std=c++20 -O3 -Icpp/include advanced_pushworld_solver.cpp \
//     cpp/src/pushworld_puzzle.cc \
//     cpp/src/heuristics/domain_transition_graph.cc \
//     cpp/src/heuristics/recursive_graph_distance.cc \
//     -o advanced_pushworld_solver
//
// Usage:
//   ./advanced_pushworld_solver macro-cut-portfolio benchmark/puzzles/level1/2\ Obstacle.pwp
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

using namespace pushworld;

namespace {

constexpr float kInf = std::numeric_limits<float>::infinity();
constexpr double kNoveltyWeight = 1000000.0;
constexpr float kDistanceEpsilon = 1e-6f;

bool SameDistance(const float a, const float b) {
  return std::fabs(a - b) <= kDistanceEpsilon;
}

bool DropsShortestDistanceByOne(const float current_distance,
                                const float next_distance) {
  return std::isfinite(current_distance) &&
         std::isfinite(next_distance) &&
         SameDistance(current_distance, next_distance + 1.0f);
}

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

struct ObjectTargetKey {
  int object_id = -1;
  Position2D current_position = 0;
  Position2D target_position = 0;

  bool operator==(const ObjectTargetKey& other) const {
    return object_id == other.object_id &&
           current_position == other.current_position &&
           target_position == other.target_position;
  }
};

struct ObjectTargetKeyHash {
  std::size_t operator()(const ObjectTargetKey& key) const {
    std::size_t seed = 0;
    boost::hash_combine(seed, key.object_id);
    boost::hash_combine(seed, key.current_position);
    boost::hash_combine(seed, key.target_position);
    return seed;
  }
};

struct CachedShortestPathDag {
  std::vector<Position2D> topo_nodes;
  std::unordered_map<Position2D, std::vector<Position2D>> outgoing;
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
    const Position2D agent_pos = state[AGENT];
    auto comp_it = agent_component_.find(agent_pos);
    if (comp_it == agent_component_.end()) return 0.0f;
    const int agent_comp = comp_it->second;

    for (int object_id = 1; object_id <= static_cast<int>(goal.size()); ++object_id) {
      const Position2D cur = state[object_id];
      const Position2D target = goal[object_id - 1];
      if (cur == target) continue;

      bool directly_accessible = false;
      for (const Position2D contact : ProgressContacts(object_id, cur, target)) {
        auto comp_contact = agent_component_.find(contact);
        if (comp_contact != agent_component_.end() && comp_contact->second == agent_comp) {
          directly_accessible = true;
          break;
        }
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
        } else if (SameDistance(rem, best_remaining)) {
          any_free_best = any_free_best || (occupied.find(nxt) == occupied.end());
        }
      }
      if (!any_free_best) penalty += 1.0f;
    }
    return penalty;
  }

  // A blocker-cut style penalty implemented as a vertex-weighted shortest-path
  // relaxation over the target object's cached shortest-path DAG. The returned
  // value is the minimum number of currently occupied cells that lie on any
  // shortest object path to its goal.
  float BlockerCutPenalty(const State& state) const {
    float penalty = 0.0f;
    const Goal& goal = puzzle_->getGoal();
    std::unordered_set<Position2D> occupied(state.begin() + 1, state.end());

    for (int object_id = 1; object_id <= static_cast<int>(goal.size()); ++object_id) {
      const Position2D cur = state[object_id];
      const Position2D target = goal[object_id - 1];
      if (cur == target) continue;

      const CachedShortestPathDag& dag = ShortestPathDag(object_id, cur, target);
      if (dag.topo_nodes.empty()) continue;

      occupied.erase(cur);
      std::unordered_map<Position2D, int> best_cost;
      best_cost.emplace(cur, 0);

      for (const Position2D u : dag.topo_nodes) {
        auto best_it = best_cost.find(u);
        if (best_it == best_cost.end()) continue;
        const int base_cost = best_it->second;
        auto edge_it = dag.outgoing.find(u);
        if (edge_it == dag.outgoing.end()) continue;

        for (const Position2D v : edge_it->second) {
          const int step_cost = occupied.find(v) != occupied.end() ? 1 : 0;
          const int next_cost = base_cost + step_cost;
          auto old = best_cost.find(v);
          if (old == best_cost.end() || next_cost < old->second) {
            best_cost[v] = next_cost;
          }
        }
      }

      occupied.insert(cur);
      auto target_it = best_cost.find(target);
      if (target_it != best_cost.end()) {
        penalty += static_cast<float>(target_it->second);
      }
    }
    return penalty;
  }

  // A contact-cut style penalty implemented as a vertex-weighted 0-1 shortest
  // path on the agent graph to any immediately progress-making contact cell.
  float ContactCutPenalty(const State& state) const {
    const Goal& goal = puzzle_->getGoal();
    const Position2D agent_pos = state[AGENT];
    const auto& agent_graph = *graphs_.at(AGENT);

    std::unordered_set<Position2D> occupied(state.begin() + 1, state.end());
    std::unordered_map<Position2D, int> best_agent_cost;
    std::deque<Position2D> dq;
    best_agent_cost.emplace(agent_pos, 0);
    dq.push_back(agent_pos);

    while (!dq.empty()) {
      const Position2D u = dq.front();
      dq.pop_front();
      const int base_cost = best_agent_cost[u];
      auto it = agent_graph.find(u);
      if (it == agent_graph.end()) continue;
      for (const Position2D v : it->second) {
        const int step_cost = occupied.find(v) != occupied.end() ? 1 : 0;
        const int next_cost = base_cost + step_cost;
        auto old = best_agent_cost.find(v);
        if (old == best_agent_cost.end() || next_cost < old->second) {
          best_agent_cost[v] = next_cost;
          if (step_cost == 0) {
            dq.push_front(v);
          } else {
            dq.push_back(v);
          }
        }
      }
    }

    float penalty = 0.0f;
    for (int object_id = 1; object_id <= static_cast<int>(goal.size()); ++object_id) {
      const Position2D cur = state[object_id];
      const Position2D target = goal[object_id - 1];
      if (cur == target) continue;

      const std::vector<Position2D> contacts = ProgressContacts(object_id, cur, target);
      float best_contact_cost = kInf;
      for (const Position2D contact : contacts) {
        auto cost_it = best_agent_cost.find(contact);
        if (cost_it == best_agent_cost.end()) continue;
        best_contact_cost = std::min(best_contact_cost,
                                     static_cast<float>(cost_it->second));
      }

      if (!std::isfinite(best_contact_cost)) {
        penalty += 1.0f;
      } else {
        penalty += best_contact_cost;
      }
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
  const CachedShortestPathDag& ShortestPathDag(const int object_id,
                                               const Position2D cur,
                                               const Position2D target) const {
    const ObjectTargetKey key{object_id, cur, target};
    auto cache_it = shortest_path_dag_cache_.find(key);
    if (cache_it != shortest_path_dag_cache_.end()) {
      return *cache_it->second;
    }

    auto dag = std::make_shared<CachedShortestPathDag>();
    const auto& object_graph = *graphs_.at(object_id);
    if (object_graph.find(cur) == object_graph.end()) {
      shortest_path_dag_cache_.emplace(key, dag);
      return *shortest_path_dag_cache_.at(key);
    }

    std::deque<Position2D> q;
    std::unordered_set<Position2D> seen;
    q.push_back(cur);
    seen.insert(cur);

    while (!q.empty()) {
      const Position2D u = q.front();
      q.pop_front();
      dag->topo_nodes.push_back(u);
      const float dist_u = path_distances_.at(object_id).getDistance(u, target);
      auto graph_it = object_graph.find(u);
      if (graph_it == object_graph.end()) continue;

      for (const Position2D v : graph_it->second) {
        const float dist_v = path_distances_.at(object_id).getDistance(v, target);
        if (!DropsShortestDistanceByOne(dist_u, dist_v)) continue;
        dag->outgoing[u].push_back(v);
        if (seen.insert(v).second) {
          q.push_back(v);
        }
      }
    }

    std::sort(dag->topo_nodes.begin(), dag->topo_nodes.end(),
              [&](const Position2D a, const Position2D b) {
                const float dist_a = path_distances_.at(object_id).getDistance(a, target);
                const float dist_b = path_distances_.at(object_id).getDistance(b, target);
                return dist_a > dist_b;
              });

    auto inserted = shortest_path_dag_cache_.emplace(key, dag);
    return *inserted.first->second;
  }

  const std::vector<Position2D>& ProgressContacts(const int object_id,
                                                  const Position2D cur,
                                                  const Position2D target) const {
    const ObjectTargetKey key{object_id, cur, target};
    auto cache_it = progress_contacts_cache_.find(key);
    if (cache_it != progress_contacts_cache_.end()) {
      return *cache_it->second;
    }

    auto contacts = std::make_shared<std::vector<Position2D>>();
    std::unordered_set<Position2D> seen_contacts;
    const auto& collisions = puzzle_->getObjectCollisions();
    const auto& agent_graph = *graphs_.at(AGENT);
    const CachedShortestPathDag& dag = ShortestPathDag(object_id, cur, target);
    auto outgoing_it = dag.outgoing.find(cur);
    if (outgoing_it != dag.outgoing.end()) {
      for (const Position2D nxt : outgoing_it->second) {
        const Position2D disp = nxt - cur;
        const auto action_it = DISPLACEMENTS_TO_ACTIONS.find(disp);
        if (action_it == DISPLACEMENTS_TO_ACTIONS.end()) continue;
        const Action action = action_it->second;
        const auto& rel_positions = collisions.dynamic_collisions[action][AGENT][object_id];
        for (const Position2D rel : rel_positions) {
          const Position2D contact = cur + rel;
          const Position2D contact_after = contact + disp;
          auto agent_node = agent_graph.find(contact);
          if (agent_node == agent_graph.end()) continue;
          if (agent_node->second.find(contact_after) == agent_node->second.end()) continue;
          if (seen_contacts.insert(contact).second) {
            contacts->push_back(contact);
          }
        }
      }
    }

    auto inserted = progress_contacts_cache_.emplace(key, contacts);
    return *inserted.first->second;
  }

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
  mutable std::unordered_map<ObjectTargetKey, std::shared_ptr<CachedShortestPathDag>, ObjectTargetKeyHash>
      shortest_path_dag_cache_;
  mutable std::unordered_map<ObjectTargetKey, std::shared_ptr<std::vector<Position2D>>, ObjectTargetKeyHash>
      progress_contacts_cache_;
  std::unordered_map<Position2D, int> agent_component_;
};

class PortfolioHeuristic {
 public:
  PortfolioHeuristic(const std::shared_ptr<PushWorldPuzzle>& puzzle,
                     bool rgd_fewest_tools)
      : puzzle_(puzzle),
        static_h_(puzzle),
        rgd_(puzzle, rgd_fewest_tools) {}

  HeuristicResult Evaluate(const RelativeState& relative_state,
                         bool use_gate_penalty = true,
                         bool use_cut_penalties = false,
                         int cut_max_unsatisfied_goals = 2) {
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

    const float h_rgd = rgd_.estimate_cost_to_goal(relative_state);

    if (use_cut_penalties &&
        CountUnsatisfiedGoals(relative_state.state) > cut_max_unsatisfied_goals) {
      use_cut_penalties = false;
    }

    if (!use_cut_penalties) {
      float h_mix = h_static + static_h_.ImmediateBlockerPenalty(relative_state.state);
      if (use_gate_penalty) {
        h_mix += static_h_.GatePenalty(relative_state.state);
      }
      out.h = std::max(h_rgd, h_mix);
      return out;
    }

    const float blocker_signal =
        static_h_.BlockerCutPenalty(relative_state.state);
    float contact_signal = 0.0f;
    if (use_gate_penalty) {
      contact_signal = static_h_.ContactCutPenalty(relative_state.state);
    }

    // The cut-style penalties are stronger replacements for the one-step
    // blocker/gate checks, so combine them conservatively with max rather than
    // summing them.
    const float h_mix = h_static + std::max(blocker_signal, contact_signal);
    out.h = std::max(h_rgd, h_mix);
    return out;
  }

 private:
  int CountUnsatisfiedGoals(const State& state) const {
    const Goal& goal = puzzle_->getGoal();
    int count = 0;
    for (int object_id = 1; object_id <= static_cast<int>(goal.size()); ++object_id) {
      if (state[object_id] != goal[object_id - 1]) {
        ++count;
      }
    }
    return count;
  }

  std::shared_ptr<PushWorldPuzzle> puzzle_;
  StaticDistanceHeuristic static_h_;
  heuristic::RecursiveGraphDistanceHeuristic rgd_;
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
  bool use_cut_penalties = false;
  int cut_max_unsatisfied_goals = 2;
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
  PortfolioHeuristic heuristic(puzzle, cfg.rgd_fewest_tools);
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
  HeuristicResult root_eval = heuristic.Evaluate(root_rs, cfg.use_gate_penalty,
                                              cfg.use_cut_penalties,
                                              cfg.cut_max_unsatisfied_goals);
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

      HeuristicResult eval = heuristic.Evaluate(succ.relative_state, cfg.use_gate_penalty,
                                             cfg.use_cut_penalties,
                                             cfg.cut_max_unsatisfied_goals);
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
  } else if (mode == "macro-portfolio") {
    cfg.style = SearchStyle::kMacro;
    cfg.rgd_fewest_tools = true;
    cfg.weight = 1.6f;
    cfg.use_gate_penalty = true;
  } else if (mode == "macro-portfolio-safe") {
    cfg.style = SearchStyle::kMacro;
    cfg.rgd_fewest_tools = true;
    cfg.weight = 1.25f;
    cfg.use_gate_penalty = true;
  } else if (mode == "macro-cut-portfolio") {
    cfg.style = SearchStyle::kMacro;
    cfg.rgd_fewest_tools = true;
    cfg.weight = 1.45f;
    cfg.use_gate_penalty = true;
    cfg.use_cut_penalties = true;
  } else if (mode == "macro-cut-portfolio-safe") {
    cfg.style = SearchStyle::kMacro;
    cfg.rgd_fewest_tools = true;
    cfg.weight = 1.20f;
    cfg.use_gate_penalty = true;
    cfg.use_cut_penalties = true;
  } else {
    throw std::invalid_argument(
        "Unknown mode. Expected one of: primitive-rgd, macro-rgd, macro-rgd-deep, macro-portfolio, macro-portfolio-safe, macro-cut-portfolio, macro-cut-portfolio-safe");
  }
  return cfg;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3 && argc != 4) {
      std::cerr << "Usage: advanced_pushworld_solver <mode> <puzzle.pwp> [time_limit_seconds]\n"
                   "Modes:\n"
                   "  primitive-rgd        Primitive weighted A* with fast RGD.\n"
                   "  macro-rgd            Macro weighted A* with fast RGD.\n"
                   "  macro-rgd-deep       Macro weighted A* with full-tool RGD.\n"
                   "  macro-portfolio      Recommended: novelty-weighted macro weighted A* with RGD + gate/blocker penalties.\n"
                   "  macro-portfolio-safe Same as portfolio but with lower heuristic weight.\n"
                   "  macro-cut-portfolio  RGD + blocker-cut/contact-cut penalties, combined conservatively.\n"
                   "  macro-cut-portfolio-safe Same as cut-portfolio but with lower heuristic weight.\n";
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
