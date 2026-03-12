# General imports.
from abc import ABC, abstractmethod
from typing import List, Tuple

# Project imports.
from ta_cbs import TACBSSolver
from single_agent_planner import get_location

class ExecutionManager(ABC):
    def __init__(self, my_map, starts, goals, **kwargs):
        # Initialize the execution manager with whichever parameters you need.
        pass

    @abstractmethod
    def get_next_location_for_all_agents(self) -> List[Tuple[int, int]]:
        # An iterator returning the next position for all agents.
        pass

    @abstractmethod
    def feedback_successful_agent_ids(self, agent_ids: List[int]):
        # Feedback for the agents that successfully moved.
        pass


class TACBSExecutionManager(ExecutionManager):
    def __init__(self, my_map, starts, goals, k=0, **kwargs):
        super().__init__(my_map, starts, goals, k=k)
        # Initialize the execution manager with whichever parameters you need.
        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.k = k

        self.solver = TACBSSolver(self.my_map, self.starts, self.goals, self.k)

        self.paths = self.solver.find_solution()
        self.t_agent = [0 for _ in range(len(self.paths))]

    def get_next_location_for_all_agents(self) -> List[Tuple[int, int]]:
        # Return the next location for all agents or empty list if done.
        print(self.t_agent, [len(path) - 1 for path in self.paths])
        if all(t > len(path) - 1 for t, path in zip(self.t_agent, self.paths)):
            return []

        locations = []
        for agent_id, path in enumerate(self.paths):
            locations.append(get_location(path, self.t_agent[agent_id]))
        return locations

    def feedback_successful_agent_ids(self, agent_ids: List[int]):
        for agent_id in agent_ids:
            self.t_agent[agent_id] += 1


class WorksReallyWellExecutionManager(ExecutionManager):
    def __init__(self, my_map, starts, goals, k=0, **kwargs):
        # Initialize the execution manager with whichever parameters you need.
        super().__init__(my_map, starts, goals, k=k)
        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.k = k

        self.solver = TACBSSolver(self.my_map, self.starts, self.goals, self.k)
        self.paths = self.solver.find_solution()

        # Progress along each path in "path-timestep" coordinates.
        self.agent_progress = [0 for _ in self.paths]
        self.locations = [get_location(path, 0) for path in self.paths]

        # TPG dependency table.
        # self.tpg_predecessor[a][t] = (pred_agent, pred_t) if agent a at timestep t
        # depends on that predecessor node; otherwise None.
        self.tpg_predecessor = [[None for _ in range(len(path))] for path in self.paths]
        self._build_tpg()

    def _build_tpg(self):
        """
        Build temporal precedence constraints for path vertices.
        """
        ##############################
        # TODO: Construct the TPG edges.
        #
        # Suggested approach:
        # 1) Sweep time from 0 to max path length - 1.
        # 2) Keep a dict "last_visit[loc] -> (agent_id, timestep)".
        # 3) For each (agent_id, t), if another agent was the last visitor of
        #    that cell, add a dependency into self.tpg_predecessor[agent_id][t].
        # 4) Treat consecutive waits of the same agent in one cell as a single
        #    visit so wait blocks do not generate self-dependencies.
        
        last_visit = dict()
        
        max_len = max(len(path) for path in self.paths)
        
        for t in range(max_len):
            for agent_id, path in enumerate(self.paths):
                if t >= len(path):
                    continue
                
                loc = get_location(path, t)
                
                # Skip waits
                if t > 0 and t < len(path) and get_location(path, t - 1) == loc:
                    continue
                
                if loc in last_visit:
                    pred_agent, pred_t = last_visit[loc]
                    if pred_agent != agent_id:
                        self.tpg_predecessor[agent_id][t] = (pred_agent, pred_t)
                        
                last_visit[loc] = (agent_id, t)

    def _next_non_wait_timestep(self, agent_id: int) -> int:
        """
        Return the next timestep where this agent leaves its current cell,
        or the final timestep if it never leaves.
        """
        path = self.paths[agent_id]
        curr_t = self.agent_progress[agent_id]
        curr_loc = get_location(path, curr_t)

        t = curr_t + 1
        while t < len(path) and get_location(path, t) == curr_loc:
            t += 1
        return min(t, len(path) - 1)

    def _dependency_satisfied(self, agent_id: int, target_t: int) -> bool:
        """
        Return True if the TPG predecessor (if any) is already completed.
        """
        pred = self.tpg_predecessor[agent_id][target_t]
        if pred is None:
            return True

        ##############################
        # TODO: Replace placeholder with your TPG readiness condition.
        # Hint: compare predecessor progress to predecessor timestep.
        pred_agent, pred_t = pred
        return self.agent_progress[pred_agent] > pred_t

    def get_next_location_for_all_agents(self) -> List[Tuple[int, int]]:
        """
        Get the next location for all agents.
        :return: List of tuples, each tuple is the next location for an agent at index i. Return empty list if done.
        """
        if all(self.agent_progress[a] >= len(path) - 1 for a, path in enumerate(self.paths)):
            return []

        locations = []
        for agent_id, path in enumerate(self.paths):
            curr_t = self.agent_progress[agent_id]
            curr_loc = get_location(path, curr_t)

            next_t = self._next_non_wait_timestep(agent_id)
            next_loc = get_location(path, next_t)

            can_move = next_loc != curr_loc and self._dependency_satisfied(agent_id, next_t)
            locations.append(next_loc if can_move else curr_loc)

        self.locations = locations
        return locations

    def feedback_successful_agent_ids(self, agent_ids: List[int]):
        """
        Feedback for the agents that successfully moved.
        :param agent_ids: List of agent IDs that successfully moved.
        :return: None
        """
        for agent_id in agent_ids:
            curr_t = self.agent_progress[agent_id]
            curr_loc = get_location(self.paths[agent_id], curr_t)
            next_t = self._next_non_wait_timestep(agent_id)

            if self.locations[agent_id] != curr_loc:
                self.agent_progress[agent_id] = next_t
