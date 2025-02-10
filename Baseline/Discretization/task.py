from Baseline.Discretization.all_resources_dp_discretization import AllResourcesDPDiscretization

"""
The TDM Ratio Legalization on each directed TDM edge is a separate task.
"""
class Task:
    def __init__(self, task_id, directed_tdm_edge, n_wire, net_to_conti_tdm_ratio):
        """
        :param task_id:
        :param directed_tdm_edge: directed edge (u, v)
        :param n_wire: number of physical wires on (u, v)
        :param net_to_conti_tdm_ratio: dict(net -> conti_tdm_ratio)
        """
        self.task_id = task_id
        self.directed_tdm_edge = directed_tdm_edge
        self.n_wire = n_wire
        self.net_to_conti_tdm_ratio = net_to_conti_tdm_ratio
        self.net_to_discrete_tdm_ratio = {}
        self.discretizer = AllResourcesDPDiscretization(self.net_to_conti_tdm_ratio, self.n_wire)


    def solve(self):
        """
        :return: cost: (n_wire, )
        cost[i] represents the cost associated with assigning
        all nets to the physical wires from wire 0 to wire i.
        """
        cost_array = self.discretizer.perform_dp()
        """
        The reason for returning self.discretizer.ass here is that when using multiprocessing, 
        modifications to the task object in the child process are not reflected in the parent process.
        Therefore, self.discretizer.ass is returned here for use in the parent process.
        """
        return cost_array, self.discretizer.ass



if __name__ == '__main__':
    pass









