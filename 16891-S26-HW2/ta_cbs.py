# Generic imports.
import copy
import time as timer
import heapq
import random
# Project imports.
from hungarian import hungarian_algorithm
from single_agent_planner import compute_heuristics, a_star, get_location, get_sum_of_cost
from kr_cbs import detect_first_collision_for_path_pair, detect_collisions_among_all_paths, standard_splitting, KRCBSSolver


class TACBSSolver(KRCBSSolver):
    """The high-level search of TA-CBS."""
    def __init__(self, my_map, starts, goals, k=0):
        """my_map   - list of lists specifying obstacle positions
        starts      - [(x1, y1), (x2, y2), ...] list of start locations
        goals       - [(x1, y1), (x2, y2), ...] list of goal locations
        k           - the parameter for K-Robust CBS
        """
        super().__init__(my_map, starts, goals, k)
        self.start_time = None
        self.my_map = my_map
        self.starts = starts
        self.goals = goals
        self.num_of_agents = len(goals)

        self.num_of_generated = 0
        self.num_of_expanded = 0
        self.CPU_time = 0

        self.open_list = []

        # compute heuristics for the low-level search
        self.heuristics = []
        for goal in self.goals:
            self.heuristics.append(compute_heuristics(my_map, goal))

        # The parameter for K-Robust CBS.
        self.k = k

    def find_solution(self):
        """
        Finds shortest paths and an optimal target assignment for all agents.
        """
        self.start_time = timer.time()
        # Generate the root node
        # constraints - list of constraints.
        # paths       - list of paths, one for each agent
        #             [[(x11, y11), (x12, y12), ...], [(x21, y21), (x22, y22), ...], ...]
        # collisions  - list of collisions in paths.
        # Mc          - Mc[i][j] is the cost of the shortest path (under constraints) for agent i to target j.
        root = {'cost': 0,
                'constraints': [],  # Like in CBS, a list of dictionaries, each dictionary is a constraint.
                'collisions': [],  # Like in CBS.
                'paths': [],  # The paths, one for each agent, that are planned for the optimal assignment under Mc.
                'Mc': {i: [float('inf') for g in range(len(self.goals))] for i in range(self.num_of_agents)}
                            # Dict[Int: List[Int]]
                            # Mc[i][j] is the cost of the shortest path (under constraints) for agent i to target j.
                }
        ##############################
        # Find initial paths for each agent to all targets.
        # Populate root['paths'] and root['Mc'] with the paths and costs.
        for agent_id in range(self.num_of_agents):
            for target_id in range(len(self.goals)):
                path = a_star(self.my_map, self.starts[agent_id], self.goals[target_id], self.heuristics[target_id],
                              agent_id, root['constraints'])
                if path is not None:
                    root['Mc'][agent_id][target_id] = len(path) - 1
                else:
                    root['Mc'][agent_id][target_id] = float('inf')
                    
        try:
            assignment = hungarian_algorithm(root['Mc'])
        except ValueError:
            raise BaseException('No solutions')
        
        for agent_id in range(self.num_of_agents):
            target_id = assignment[agent_id]
            path = a_star(self.my_map, self.starts[agent_id], self.goals[target_id], self.heuristics[target_id],
                          agent_id, root['constraints'])
            if path is None:
                raise BaseException('No solutions')
            root['paths'].append(path)
        
        root['cost'] = get_sum_of_cost(root['paths'])
        root['collisions'] = detect_collisions_among_all_paths(root['paths'], self.k)
        self.push_node(root)

        ##############################
        # High-Level Search
        #  Repeat the following as long as the open list is not empty:
        #    1. Get the next node from the open list (you can use self.pop_node()
        #    2. If this node has no collisions, return the solution stored in its 'paths' field.
        #    3. Otherwise, choose the first collision and convert to a list of constraints (using your
        #       standard_splitting function).
        #       For each constraint created:
        #         3a. Create a new child CT node.
        #         3b. Replan the affected agent paths to all goals and update Mc with the costs.
        #         3c. Find the new optimal assignment and paths.
        #         3d. Add the new child CT node to the open list.

        while len(self.open_list) > 0:
            P = self.pop_node()
            
            if len(P['collisions']) == 0:
                self.print_results(P)
                return P['paths']
            
            collision = P['collisions'][0]
            constraints = standard_splitting(collision)
            
            for constraint in constraints:
                Q = copy.deepcopy(P)
                Q['constraints'].append(constraint)
                agent_id = constraint['agent']
                
                for target_id in range(len(self.goals)):
                    path = a_star(self.my_map, self.starts[agent_id], self.goals[target_id], self.heuristics[target_id],
                                  agent_id, Q['constraints'])
                    if path is not None:
                        Q['Mc'][agent_id][target_id] = len(path) - 1
                    else:
                        Q['Mc'][agent_id][target_id] = float('inf')
                
                try:
                    assignment = hungarian_algorithm(Q['Mc'])
                except ValueError:
                    continue
                
                Q['paths'] = []
                feasible_paths = True
                for agent_id in range(self.num_of_agents):
                    target_id = assignment[agent_id]
                    path = a_star(self.my_map, self.starts[agent_id], self.goals[target_id], self.heuristics[target_id],
                                  agent_id, Q['constraints'])
                    if path is None:
                        feasible_paths = False
                        break
                    Q['paths'].append(path)

                if not feasible_paths:
                    continue
                
                Q['cost'] = get_sum_of_cost(Q['paths'])
                Q['collisions'] = detect_collisions_among_all_paths(Q['paths'], self.k)
                self.push_node(Q)
                
        raise BaseException('No solutions')
                
    def print_results(self, node):
        print("\n Found a solution! \n")
        CPU_time = timer.time() - self.start_time
        print("CPU time (s):    {:.2f}".format(CPU_time))
        print("Sum of costs:    {}".format(get_sum_of_cost(node['paths'])))
        print("Expanded nodes:  {}".format(self.num_of_expanded))
        print("Generated nodes: {}".format(self.num_of_generated))