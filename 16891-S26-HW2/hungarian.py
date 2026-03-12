import numpy as np
from scipy.optimize import linear_sum_assignment


def hungarian_algorithm(agent_goal_costs):
    """
    :param agent_goal_costs: A dictionary mapping agent ids to a list of costs for each goal.
                            The order of goals is the same for all agents.
    :return: A dictionary mapping agent id to goal id.
    """
    ##############################
    # Implement the Hungarian algorithm.
    # Return the optimal assignment as a dictionary mapping agent id to goal id.
    if len(agent_goal_costs) == 0:
        return {}

    agent_ids = sorted(agent_goal_costs.keys())

    # Rows as agents and columns as goals
    # Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
    cost_matrix = np.array([agent_goal_costs[agent_id] for agent_id in agent_ids], dtype=float)
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    # DEBUG
    # print(agent_ids)
    # print(row_indices)
    # print(col_indices)

    return {agent_ids[row]: int(col) for row, col in zip(row_indices, col_indices)}