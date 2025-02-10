import pickle
import numpy as np
import sys

sys.path.append('../../../')

from Source.Baseline.Discretization.task import Task
from Source.Baseline.ContiTDMSolver.solver_mixin import SolverMixin
import multiprocessing as mp
from Source.Baseline.Discretization.dp_legal_mixin import DPLegalMixIn

def worker(task: Task):
    """
    :param task:
    :return: (Cost array, Assignment Matrix)
    Cost array: cost for 1, 2, 3, ..., n wires
    Assignment Scheme: The Assignment Scheme is a matrix where Ass[i, j] represents the number of nets assigned to
    the j-th physical line when nets numbered 0 to i are allocated to physical lines numbered 0 to j.
    """
    if task is None:
        return None, None
    print(f"Task {task.task_id} has been finished~")
    return task.solve()


class AllResourcesDPLegal(DPLegalMixIn):
    def __init__(self,
                 nets,
                 weighted_routing_resource_net,
                 dir_tdm_edge_path_tdm_ratio_matrix,
                 directed_tdm_edge_row_idx_map,
                 enable_multiprocessing=True,
                 n_processes=-1):

        """
        :param nets:
        :param weighted_routing_resource_net:
        Attributes of an edge:
          [1] resource_cnt : number of physical wires on the edge
          [2] type: 0/1 represents internal/external edge
          [3] weight: delay of the edge
          [4] pre_adjusted_weight: weight + 1/resource_cnt for TDM edge; weight for non-TDM edge.
          [5] usage: the number of nets using the edge
        :param dir_tdm_edge_path_tdm_ratio_matrix:
        :param directed_tdm_edge_row_idx_map:
        :param enable_multiprocessing:
        :param n_processes: -1 indicates using cpu_count() to determine the number of processes to start.
        """
        self.nets = nets
        self.weighted_routing_resource_net = weighted_routing_resource_net
        self.dir_tdm_edge_path_tdm_ratio_matrix = dir_tdm_edge_path_tdm_ratio_matrix
        self.directed_tdm_edge_to_row_idx = directed_tdm_edge_row_idx_map
        self.enable_multiprocessing = enable_multiprocessing

        if n_processes == -1:
            self.n_processes = mp.cpu_count()
        else:
            self.n_processes = n_processes
        """
        (u, v) -> routing net set
        """
        self.directed_tdm_edge_to_routing_net_set = SolverMixin.get_directed_tdm_edge_to_routing_net_set(self.nets,
                                                                                                         self.weighted_routing_resource_net)
        """
        dict(net -> base_col_idx)
        """
        self.net_to_base_col_idx_map = self.get_net_to_base_col_index_map()

        """
        (n_ttl_path, )
        """
        self.non_tdm_delay_array = SolverMixin.get_non_tdm_edge_delay_array(self.nets,
                                                                            self.weighted_routing_resource_net)

        """
        (n_ttl_path, )
        """
        self.n_tdm_edge_array = SolverMixin.get_n_tdm_edge_array(self.nets,
                                                                 self.weighted_routing_resource_net)

        system_delay = self.get_system_delay()
        print(f'Before legalization, System delay is {system_delay}.')




    def solve_tdm_edge(self, param):
        """
         Perform legalization for nets on an FPGA pair (u, v).
        :param param is a tuple of (u, v, task_id)
        :return:
        """
        u, v, task_id = param
        assert self.weighted_routing_resource_net[u][v]['type'] == 1
        tasks = self.set_up_tdm_edge_tasks(u, v, task_id)

        cost_array_0, assignment_0 = worker(tasks[0])
        cost_array_1, assignment_1 = worker(tasks[1])

        n_wire = tasks[0].n_wire

        if cost_array_0 is not None and cost_array_1 is not None:
            """
            No directed edge can monopolize resources.
            Thus the last element of the cost array is removed.
            """
            clipped_cost_array_0, reversed_clipped_cost_array_1 = cost_array_0[:-1], cost_array_1[:-1][::-1]

            n_wire_0 = np.argmin(np.maximum(clipped_cost_array_0, reversed_clipped_cost_array_1)) + 1

        if cost_array_0 is None:
            n_wire_0 = 0

        if cost_array_1 is None:
            n_wire_0 = n_wire

        n_wire_1 = n_wire - n_wire_0

        net_to_discrete_tdm_ratio_task_0 = tasks[0].discretizer.construct_assignment_scheme(n_wire_0)
        net_to_discrete_tdm_ratio_task_1 = tasks[1].discretizer.construct_assignment_scheme(n_wire_1)

        row_idx_0 = self.directed_tdm_edge_to_row_idx[tasks[0].directed_tdm_edge]
        row_idx_1 = self.directed_tdm_edge_to_row_idx[tasks[1].directed_tdm_edge]

        res_array_0 = self.gen_res_array(tasks[0].directed_tdm_edge, net_to_discrete_tdm_ratio_task_0)
        res_array_1 = self.gen_res_array(tasks[1].directed_tdm_edge, net_to_discrete_tdm_ratio_task_1)

        return row_idx_0, row_idx_1, res_array_0, res_array_1




    def set_up_tdm_edge_tasks(self, u, v, task_id):
        """
        :param u:
        :param v:
        :param task_id:
        :return:
        """
        data = self.weighted_routing_resource_net[u][v]
        resource_cnt = data['resource_cnt']
        tasks = []
        for e in [(u, v), (v, u)]:
            net_to_conti_tdm_ratio = {}
            row_idx = self.directed_tdm_edge_to_row_idx[e]

            """
            If the edge e is not in directed_tdm_edge_to_routing_net_set, it indicates that 
            no net uses e.
            """
            if e in self.directed_tdm_edge_to_routing_net_set:
                for net in self.directed_tdm_edge_to_routing_net_set[e]:
                    base_col_idx = self.net_to_base_col_idx_map[net]
                    """
                    A net may have multiple pins, but not all of them pass through edge e. 
                    Therefore, it is necessary to identify the first pin corresponding to edge e.
                    """
                    offset = net.directed_tdm_edge_to_sink_idx_list[e][0]
                    conti_tdm_ratio = self.dir_tdm_edge_path_tdm_ratio_matrix[row_idx, base_col_idx+offset]
                    net_to_conti_tdm_ratio[net] = conti_tdm_ratio
                task = Task(task_id, e, resource_cnt, net_to_conti_tdm_ratio)
                tasks.append(task)
            else:
                tasks.append(None)

        return tasks








if __name__ == '__main__':
    # HP Server
    # Lenovo Server
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase4/run_20241113111933/run_20241120170743/run_20241126195408/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase5/run_20241113111752/run_20241120170927/run_20241126200029/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase6/run_20241111193533/run_20241119093605/run_20241126174313/conti_tdm_ratio.pkl"
    conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase7/run_20241112201514/run_20241119141119/run_20241125164618/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase8/run_20241113093433/run_20241119165208/run_20241126170653/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase9/run_20241113112804/run_20241119170429/run_20241126182817/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase10/run_20241113114751/run_20241120144955/run_20241126204916/conti_tdm_ratio.pkl"


    with open(conti_sol_res, "rb") as f:
        nets, weighted_routing_resource_net, dir_tdm_edge_path_tdm_ratio_matrix, directed_tdm_edge_row_idx_map = pickle.load(f)

    dp_legalizer = AllResourcesDPLegal(nets, weighted_routing_resource_net, dir_tdm_edge_path_tdm_ratio_matrix, directed_tdm_edge_row_idx_map, n_processes=16)

    dp_legalizer.perform_dp_legal()














