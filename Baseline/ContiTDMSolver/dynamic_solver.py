import copy
import pickle
import time
import math

import numpy as np
from Baseline.ContiTDMSolver.solver_mixin import SolverMixin


class DynamicSolver(SolverMixin):
    def perf_init_assign(self):
        """
        The net direction is ignored during the process of finding the initial solution.
        :return:
        """
        """
        dict(matrix_idx -> tdm_ratio)
        matrix_idx is the index of the directed_tdm_edge_to_routing_net_set.
        """

        for u, v, data in self.weighted_routing_resource_net.edges(data=True):
            if data['type'] == 1:
                """
                All nets from u to v or from v to u.
                """
                net_lst = (list(self.directed_tdm_edge_to_routing_net_set[(u, v)])
                                   + list(self.directed_tdm_edge_to_routing_net_set[(v, u)]))

                ttl_net_delay = sum([net.criticality for net in net_lst])


                for e in [(u, v), (v, u)]:
                    row_idx = self.directed_tdm_edge_row_idx_map[e]

                    """
                    If e not in the dict, it means that e is not used by any net.
                    """
                    if e not in self.directed_tdm_edge_to_routing_net_set:
                        continue

                    for net in self.directed_tdm_edge_to_routing_net_set[e]:
                        col_base_idx = self.net_to_base_col_idx_map[net]
                        net_e_tdm_ratio = ttl_net_delay / (data['resource_cnt'] * net.criticality)

                        for off_set in net.directed_tdm_edge_to_sink_idx_list[e]:
                            col_idx = col_base_idx + off_set
                            matrix_idx = row_idx, col_idx
                            self.dir_tdm_edge_path_tdm_ratio_matrix[matrix_idx] = net_e_tdm_ratio

                delay = self.get_delay()

                """
                Dynamically update the net delay.
                """
                self.update_criticality(delay, net_lst)

                self.system_delay = max(delay)
                print(f'Delay: {self.system_delay}\tMin Delay: {self.min_system_delay}\tGap: '
                      f'{round(self.system_delay - self.min_system_delay, 4)}')

                if self.system_delay < self.min_system_delay:
                    self.min_system_delay = self.system_delay
                    self.best_dir_tdm_edge_path_tdm_ratio_matrix = copy.deepcopy(self.dir_tdm_edge_path_tdm_ratio_matrix)




    def perf_init_assign_and_refine(self):

        self.perf_init_assign()
        stagnation_rounds = 0
        while True:
            for u, v, data in self.weighted_routing_resource_net.edges(data=True):
                if data['type'] == 1:
                    """
                    If the edge (u, v) is not in directed_tdm_edge_to_routing_net_set, 
                    it indicates that no net uses the edge (u, v).
                    """
                    ttl_criticality_u_v = sum([net.criticality for net in self.directed_tdm_edge_to_routing_net_set[(u, v)]])
                    ttl_criticality_v_u = sum([net.criticality for net in self.directed_tdm_edge_to_routing_net_set[(v, u)]])

                    if ttl_criticality_u_v > 0 or ttl_criticality_v_u > 0:
                        resource_u_v = data['resource_cnt'] * (ttl_criticality_u_v / (ttl_criticality_u_v + ttl_criticality_v_u))
                        resource_v_u = data['resource_cnt'] - resource_u_v
                        if ttl_criticality_u_v > 0:
                            self.refine(u, v, resource_u_v)
                        if ttl_criticality_v_u > 0:
                            self.refine(v, u, resource_v_u)

                    delay = self.get_delay()
                    self.update_criticality(delay, self.directed_tdm_edge_to_routing_net_set[(u, v)])
                    self.update_criticality(delay, self.directed_tdm_edge_to_routing_net_set[(v, u)])

            delay = self.get_delay()
            self.system_delay = max(delay)
            print(f'Delay: {self.system_delay}\tMin Delay: {self.min_system_delay}\tGap: '
                  f'{round(self.system_delay - self.min_system_delay, 4)}')

            if self.system_delay < self.min_system_delay - self.sig_thresh:
                self.min_system_delay = self.system_delay
                self.best_dir_tdm_edge_path_tdm_ratio_matrix = copy.deepcopy(self.dir_tdm_edge_path_tdm_ratio_matrix)
                stagnation_rounds = 0
            else:
                stagnation_rounds += 1

            if stagnation_rounds >= self.stag_rounds_thresh:
                break








if __name__ == '__main__':
    # HP Server
    # two_stage_reroute_res = "../../Res/Baseline/testcase6/run_20241117152129/run_20241117160605/two_stage_reroute_res.pkl"
    # Lenovo Server
    # two_stage_reroute_res = "../../Res/Baseline/testcase4/run_20241113111933/run_20241120170743/two_stage_reroute_res.pkl"
    # two_stage_reroute_res = "../../Res/Baseline/testcase5/run_20241113111752/run_20241120170927/two_stage_reroute_res.pkl"
    # two_stage_reroute_res = "../../Res/Baseline/testcase6/run_20241111193533/run_20241119093605/two_stage_reroute_res.pkl"
    two_stage_reroute_res = "../../Res/Baseline/testcase7/run_20241112201514/run_20241119141119/two_stage_reroute_res.pkl"
    # two_stage_reroute_res = "../../Res/Baseline/testcase8/run_20241113093433/run_20241119165208/two_stage_reroute_res.pkl"
    # two_stage_reroute_res = "../../Res/Baseline/testcase9/run_20241113112804/run_20241119170429/two_stage_reroute_res.pkl"
    # two_stage_reroute_res = "../../Res/Baseline/testcase10/run_20241113114751/run_20241120144955/two_stage_reroute_res.pkl"



    with open(two_stage_reroute_res, "rb") as f:
        nets, weighted_routing_resource_net, _ = pickle.load(f)

    dynamic_solver = DynamicSolver(nets, weighted_routing_resource_net)
    dynamic_solver.perf_init_assign_and_refine()

