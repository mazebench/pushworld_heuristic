#ifndef PUSHWORLD_RECURSIVE_SUBSET_PROBES_HPP_
#define PUSHWORLD_RECURSIVE_SUBSET_PROBES_HPP_

#include <algorithm>
#include <climits>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <queue>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pushworld_puzzle.h"

namespace pushworld {
namespace recursive_subgoals {

// Callback used to solve a reduced PushWorld puzzle. The returned string, when
// present, is expected to be a primitive UDLR plan.
using SolverCallback =
    std::function<std::optional<std::string>(const std::shared_ptr<PushWorldPuzzle>&)>;

struct ReducedSolveResult {
  bool solved = false;
  int plan_length = INT_MAX;
  std::vector<int> object_ids;  // Extra objects included or removed.
};

struct BlockerScore {
  int object_id = -1;
  int baseline_plan_length = INT_MAX;
  int deleted_plan_length = INT_MAX;
  int score = INT_MIN;  // Higher means more likely to be a blocker.
};

// A sliced PushWorld puzzle containing the agent, one or more focus goal
// objects, and an arbitrary set of extra objects. Focus goal objects always
// come immediately after the agent so that their goal positions can occupy the
// first entries of Goal, as required by PushWorldPuzzle.
struct SubsetPuzzleTemplate {
  std::vector<int> sub_to_orig;
  std::vector<int> orig_to_sub;
  Goal sub_goal;
  ObjectCollisions sub_collisions;

  State ProjectState(const State& full_state) const {
    State sub_state(sub_to_orig.size());
    for (std::size_t i = 0; i < sub_to_orig.size(); ++i) {
      sub_state[i] = full_state[sub_to_orig[i]];
    }
    return sub_state;
  }

  std::shared_ptr<PushWorldPuzzle> MakePuzzle(const State& full_state) const {
    return std::make_shared<PushWorldPuzzle>(ProjectState(full_state), sub_goal,
                                             sub_collisions);
  }
};

inline std::vector<int> SortedUniqueObjectIds(const std::vector<int>& ids,
                                              const int num_objects) {
  std::vector<int> out;
  out.reserve(ids.size());
  std::vector<bool> seen(num_objects, false);
  for (int id : ids) {
    if (id < 0 || id >= num_objects) {
      throw std::out_of_range("Object id out of range.");
    }
    if (!seen[id]) {
      seen[id] = true;
      out.push_back(id);
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

inline SubsetPuzzleTemplate BuildSubsetPuzzleTemplate(
    const PushWorldPuzzle& full_puzzle,
    const std::vector<int>& focus_goal_object_ids,
    const std::vector<int>& extra_object_ids) {
  const State& full_initial = full_puzzle.getInitialState();
  const Goal& full_goal = full_puzzle.getGoal();
  const ObjectCollisions& full_collisions = full_puzzle.getObjectCollisions();
  const int num_objects = static_cast<int>(full_initial.size());

  std::vector<int> focus = SortedUniqueObjectIds(focus_goal_object_ids, num_objects);
  std::vector<int> extras = SortedUniqueObjectIds(extra_object_ids, num_objects);

  // The agent is always present as object 0. Goals must refer to original goal
  // objects, which in PushWorld are exactly object ids 1..goal.size().
  std::vector<bool> used(num_objects, false);
  std::vector<int> sub_to_orig;
  sub_to_orig.reserve(1 + focus.size() + extras.size());
  sub_to_orig.push_back(AGENT);
  used[AGENT] = true;

  for (int id : focus) {
    if (id == AGENT) continue;
    if (id < 1 || id > static_cast<int>(full_goal.size())) {
      throw std::invalid_argument(
          "Focus goal ids must correspond to original goal objects.");
    }
    if (!used[id]) {
      used[id] = true;
      sub_to_orig.push_back(id);
    }
  }
  for (int id : extras) {
    if (id == AGENT) continue;
    if (!used[id]) {
      used[id] = true;
      sub_to_orig.push_back(id);
    }
  }

  SubsetPuzzleTemplate tmpl;
  tmpl.sub_to_orig = sub_to_orig;
  tmpl.orig_to_sub.assign(num_objects, -1);
  for (int sub_id = 0; sub_id < static_cast<int>(sub_to_orig.size()); ++sub_id) {
    tmpl.orig_to_sub[sub_to_orig[sub_id]] = sub_id;
  }

  tmpl.sub_goal.resize(focus.size());
  for (std::size_t i = 0; i < focus.size(); ++i) {
    const int orig_goal_object_id = focus[i];
    tmpl.sub_goal[i] = full_goal[orig_goal_object_id - 1];
  }

  tmpl.sub_collisions.resize(sub_to_orig.size());
  for (int action = 0; action < NUM_ACTIONS; ++action) {
    for (int sub_i = 0; sub_i < static_cast<int>(sub_to_orig.size()); ++sub_i) {
      const int orig_i = sub_to_orig[sub_i];
      tmpl.sub_collisions.static_collisions[action][sub_i] =
          full_collisions.static_collisions[action][orig_i];
      for (int sub_j = 0; sub_j < static_cast<int>(sub_to_orig.size()); ++sub_j) {
        const int orig_j = sub_to_orig[sub_j];
        tmpl.sub_collisions.dynamic_collisions[action][sub_i][sub_j] =
            full_collisions.dynamic_collisions[action][orig_i][orig_j];
      }
    }
  }
  return tmpl;
}

class ToolAffordanceGraph {
 public:
  explicit ToolAffordanceGraph(const PushWorldPuzzle& puzzle) {
    const int num_objects = static_cast<int>(puzzle.getInitialState().size());
    can_push_.assign(num_objects, {});
    can_be_pushed_by_.assign(num_objects, {});
    const ObjectCollisions& collisions = puzzle.getObjectCollisions();
    for (int pusher = 0; pusher < num_objects; ++pusher) {
      for (int pushee = 1; pushee < num_objects; ++pushee) {
        if (pusher == pushee) continue;
        bool ever_pushes = false;
        for (int action = 0; action < NUM_ACTIONS && !ever_pushes; ++action) {
          ever_pushes = !collisions.dynamic_collisions[action][pusher][pushee].empty();
        }
        if (ever_pushes) {
          can_push_[pusher].push_back(pushee);
          can_be_pushed_by_[pushee].push_back(pusher);
        }
      }
    }
  }

  bool CanEverPush(const int pusher, const int pushee) const {
    const auto& nbrs = can_push_.at(pusher);
    return std::find(nbrs.begin(), nbrs.end(), pushee) != nbrs.end();
  }

  // Reverse-BFS from the target object through the affordance graph.
  std::vector<int> CandidateToolsForTarget(const int target_object_id,
                                           const int max_depth,
                                           const std::size_t candidate_cap = 12) const {
    std::vector<int> result;
    if (max_depth <= 0) return result;

    std::queue<std::pair<int, int>> q;
    std::vector<bool> visited(can_push_.size(), false);
    q.push({target_object_id, 0});
    visited[target_object_id] = true;

    while (!q.empty()) {
      const auto [object_id, depth] = q.front();
      q.pop();
      if (depth == max_depth) continue;
      for (int pusher : can_be_pushed_by_[object_id]) {
        if (pusher == AGENT) continue;
        if (!visited[pusher]) {
          visited[pusher] = true;
          result.push_back(pusher);
          q.push({pusher, depth + 1});
          if (result.size() >= candidate_cap) {
            return result;
          }
        }
      }
    }
    return result;
  }

 private:
  std::vector<std::vector<int>> can_push_;
  std::vector<std::vector<int>> can_be_pushed_by_;
};

inline void EnumerateFixedSizeSubsets(
    const std::vector<int>& values,
    const int k,
    const std::function<void(const std::vector<int>&)>& callback) {
  if (k < 0 || k > static_cast<int>(values.size())) return;
  std::vector<int> subset;
  subset.reserve(k);
  std::function<void(int, int)> dfs = [&](int start, int left) {
    if (left == 0) {
      callback(subset);
      return;
    }
    for (int i = start; i <= static_cast<int>(values.size()) - left; ++i) {
      subset.push_back(values[i]);
      dfs(i + 1, left - 1);
      subset.pop_back();
    }
  };
  dfs(0, k);
}

// "Addition recursion": start with the goal object alone and add a small set
// of extra objects until the reduced puzzle becomes solvable. Smallest solved
// subsets tend to be tools or tool components.
inline ReducedSolveResult FindMinimalToolSet(
    const PushWorldPuzzle& full_puzzle,
    const State& state,
    const int focus_goal_object_id,
    const SolverCallback& solve_reduced,
    const int max_extra_tools = 2,
    const int affordance_depth = 2,
    const std::size_t candidate_cap = 10) {
  ToolAffordanceGraph affordances(full_puzzle);
  std::vector<int> candidates = affordances.CandidateToolsForTarget(
      focus_goal_object_id, affordance_depth, candidate_cap);

  ReducedSolveResult best;
  best.object_ids.clear();

  for (int k = 0; k <= max_extra_tools; ++k) {
    bool found_at_this_k = false;
    EnumerateFixedSizeSubsets(
        candidates, k, [&](const std::vector<int>& extras) {
          if (found_at_this_k) return;
          SubsetPuzzleTemplate tmpl = BuildSubsetPuzzleTemplate(
              full_puzzle, {focus_goal_object_id}, extras);
          std::shared_ptr<PushWorldPuzzle> sub_puzzle = tmpl.MakePuzzle(state);
          std::optional<std::string> plan = solve_reduced(sub_puzzle);
          if (plan.has_value()) {
            found_at_this_k = true;
            best.solved = true;
            best.plan_length = static_cast<int>(plan->size());
            best.object_ids = extras;
          }
        });
    if (found_at_this_k) return best;
  }
  return best;
}

// "Deletion recursion": keep a focus goal plus all currently relevant objects,
// then delete one object at a time. Objects whose deletion suddenly makes the
// focus problem solvable, or much shorter, are strong blocker candidates.
inline std::vector<BlockerScore> FindDeletionBlockers(
    const PushWorldPuzzle& full_puzzle,
    const State& state,
    const int focus_goal_object_id,
    const SolverCallback& solve_reduced,
    const std::vector<int>& always_keep = {},
    const int candidate_cap = -1) {
  const int num_objects = static_cast<int>(full_puzzle.getInitialState().size());
  std::vector<int> extras;
  extras.reserve(num_objects);
  std::vector<bool> keep(num_objects, false);
  keep[AGENT] = true;
  keep[focus_goal_object_id] = true;
  for (int id : always_keep) {
    if (id >= 0 && id < num_objects) keep[id] = true;
  }
  for (int id = 1; id < num_objects; ++id) {
    if (id == focus_goal_object_id) continue;
    extras.push_back(id);
  }

  SubsetPuzzleTemplate baseline_tmpl =
      BuildSubsetPuzzleTemplate(full_puzzle, {focus_goal_object_id}, extras);
  std::optional<std::string> baseline_plan =
      solve_reduced(baseline_tmpl.MakePuzzle(state));
  const int baseline_len = baseline_plan.has_value() ? static_cast<int>(baseline_plan->size())
                                                     : INT_MAX;

  std::vector<BlockerScore> scores;
  for (int deleted_id : extras) {
    if (keep[deleted_id]) continue;
    std::vector<int> reduced_extras;
    reduced_extras.reserve(extras.size());
    for (int id : extras) {
      if (id != deleted_id) reduced_extras.push_back(id);
    }
    SubsetPuzzleTemplate reduced_tmpl = BuildSubsetPuzzleTemplate(
        full_puzzle, {focus_goal_object_id}, reduced_extras);
    std::optional<std::string> deleted_plan =
        solve_reduced(reduced_tmpl.MakePuzzle(state));

    BlockerScore score;
    score.object_id = deleted_id;
    score.baseline_plan_length = baseline_len;
    score.deleted_plan_length = deleted_plan.has_value()
                                    ? static_cast<int>(deleted_plan->size())
                                    : INT_MAX;
    if (baseline_len == INT_MAX && score.deleted_plan_length == INT_MAX) {
      score.score = INT_MIN / 4;
    } else if (baseline_len == INT_MAX) {
      score.score = 1000000 - score.deleted_plan_length;
    } else if (score.deleted_plan_length == INT_MAX) {
      score.score = INT_MIN / 8;
    } else {
      score.score = baseline_len - score.deleted_plan_length;
    }
    scores.push_back(score);
  }

  std::sort(scores.begin(), scores.end(),
            [](const BlockerScore& a, const BlockerScore& b) {
              if (a.score != b.score) return a.score > b.score;
              if (a.deleted_plan_length != b.deleted_plan_length) {
                return a.deleted_plan_length < b.deleted_plan_length;
              }
              return a.object_id < b.object_id;
            });

  if (candidate_cap >= 0 && static_cast<int>(scores.size()) > candidate_cap) {
    scores.resize(candidate_cap);
  }
  return scores;
}

// A simple recursive estimate that combines the two probes above.
//
// 1. Find the smallest tool set that makes the isolated focus goal solvable.
// 2. Estimate the cost of that reduced solution.
// 3. Probe one-object deletions in the larger focus problem to identify likely
//    blockers, and add a small penalty for the top blockers.
//
// The returned cost is intended as an inadmissible ranking heuristic, not as a
// proof of optimality.
inline float RecursiveSubsetEstimate(
    const PushWorldPuzzle& full_puzzle,
    const State& state,
    const int focus_goal_object_id,
    const SolverCallback& solve_reduced,
    const int max_extra_tools = 2,
    const int affordance_depth = 2,
    const std::size_t candidate_cap = 10,
    const int blocker_probe_cap = 3,
    const float blocker_penalty = 1.5f) {
  ReducedSolveResult tools = FindMinimalToolSet(
      full_puzzle, state, focus_goal_object_id, solve_reduced, max_extra_tools,
      affordance_depth, candidate_cap);

  float estimate = 0.0f;
  if (tools.solved) {
    estimate += static_cast<float>(tools.plan_length);
  } else {
    // If even the isolated subproblem is unsolved by the reduced planner, add
    // a penalty that grows with the number of currently available candidates.
    estimate += 10.0f + static_cast<float>(candidate_cap);
  }

  std::vector<BlockerScore> blockers = FindDeletionBlockers(
      full_puzzle, state, focus_goal_object_id, solve_reduced, tools.object_ids,
      blocker_probe_cap);

  int useful_blockers = 0;
  for (const BlockerScore& b : blockers) {
    if (b.score > 0 || (b.baseline_plan_length == INT_MAX &&
                        b.deleted_plan_length != INT_MAX)) {
      ++useful_blockers;
    }
  }
  estimate += blocker_penalty * useful_blockers;
  return estimate;
}

}  // namespace recursive_subgoals
}  // namespace pushworld

#endif  // PUSHWORLD_RECURSIVE_SUBSET_PROBES_HPP_
