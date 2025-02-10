import pickle
import sys


sys.path.append('../../')

from Baseline.ContiTDMSolver.solver_mixin import SolverMixin
import multiprocessing as mp
from Baseline.Discretization.incremental_dp_discretization import IncrementalDPDiscretization
from Baseline.Discretization.dp_legal_mixin import DPLegalMixIn

class IncrementalDPLegal(DPLegalMixIn):
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

        n_wire = self.weighted_routing_resource_net[u][v]['resource_cnt']

        e_list = [(u, v), (v, u)]

        net_to_conti_tdm_ratio_list = []
        for e in e_list:
            net_to_conti_tdm_ratio_list.append(self.get_net_to_conti_tdm_ratio(*e))

        discretizer_lst = []
        for net_to_conti_tdm_ratio in net_to_conti_tdm_ratio_list:
            discretizer_lst.append(IncrementalDPDiscretization(net_to_conti_tdm_ratio, n_wire))

        """
        Allocate resources to the side with the higher cost, until resources are exhausted.
        cost[i]: the objective value obtained by assigning net 0 ~ net i.
        """
        while discretizer_lst[0].n_allocated_resources + discretizer_lst[1].n_allocated_resources < n_wire:
            if discretizer_lst[0].cost[-1] <= discretizer_lst[1].cost[-1]:
                discretizer_lst[1].increase_resource_and_update_cost()
            else:
                discretizer_lst[0].increase_resource_and_update_cost()

        net_to_discrete_tdm_ratio_lst = []
        for discretizer in discretizer_lst:
            if discretizer.n_allocated_resources > 0:
                net_to_discrete_tdm_ratio = discretizer.construct_assignment_scheme(discretizer.n_allocated_resources)
                net_to_discrete_tdm_ratio_lst.append(net_to_discrete_tdm_ratio)
            else:
                net_to_discrete_tdm_ratio_lst.append(None)


        row_idx_0 = self.directed_tdm_edge_to_row_idx[e_list[0]]
        row_idx_1 = self.directed_tdm_edge_to_row_idx[e_list[1]]

        res_array_0 = self.gen_res_array(e_list[0], net_to_discrete_tdm_ratio_lst[0])
        res_array_1 = self.gen_res_array(e_list[1], net_to_discrete_tdm_ratio_lst[1])

        return row_idx_0, row_idx_1, res_array_0, res_array_1




    def get_net_to_conti_tdm_ratio(self, u, v):
        """
        :param u:
        :param v:
        :return: dict(net -> continuous tdm ratio on the directed tdm edge (u, v))
        """

        e = (u, v)
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

        return net_to_conti_tdm_ratio












if __name__ == '__main__':
    # HP Server
    # Lenovo Server
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase4/run_20241113111933/run_20241120170743/run_20241126195408/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase5/run_20241113111752/run_20241120170927/run_20241126200029/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase6/run_20241111193533/run_20241119093605/run_20241126174313/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase7/run_20241112201514/run_20241119141119/run_20241125164618/conti_tdm_ratio.pkl"
    conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase8/run_20241113093433/run_20241119165208/run_20241126170653/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase9/run_20241113112804/run_20241119170429/run_20241126182817/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase10/run_20241113114751/run_20241120144955/run_20241126204916/conti_tdm_ratio.pkl"


    with open(conti_sol_res, "rb") as f:
        nets, weighted_routing_resource_net, dir_tdm_edge_path_tdm_ratio_matrix, directed_tdm_edge_row_idx_map = pickle.load(f)

    dp_legalizer = IncrementalDPLegal(nets, weighted_routing_resource_net, dir_tdm_edge_path_tdm_ratio_matrix, directed_tdm_edge_row_idx_map, n_processes=16)

    dp_legalizer.perform_dp_legal()














