import copy
import pickle

from Source.Baseline.ContiTDMSolver.solver_mixin import SolverMixin


class Solver(SolverMixin):

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
                net_lst = (list(self.directed_tdm_edge_to_routing_net_set.get((u, v), []))
                                   + list(self.directed_tdm_edge_to_routing_net_set.get((v, u), [])))

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
                self.system_delay = max(delay)
                print(f'Delay: {self.system_delay}\tMin Delay: {self.min_system_delay}\tGap: '
                      f'{round(self.system_delay - self.min_system_delay, 4)}')

                if self.system_delay < self.min_system_delay:
                    self.min_system_delay = self.system_delay
                    self.best_dir_tdm_edge_path_tdm_ratio_matrix = copy.deepcopy(self.dir_tdm_edge_path_tdm_ratio_matrix)


        """
        Update net delay.
        """
        delay = self.get_delay()
        self.update_criticality(delay)



    def perf_init_assign_and_refine(self):

        print("Perform Initial Assignment")
        self.perf_init_assign()
        print("Finish Initial Assignment")

        while True:
            for u, v, data in self.weighted_routing_resource_net.edges(data=True):
                if data['type'] == 1:
                    """
                    If the edge (u, v) is not in directed_tdm_edge_to_routing_net_set, 
                    it indicates that no net uses the edge (u, v).
                    """
                    ttl_criticality_u_v = sum([net.criticality for net in self.directed_tdm_edge_to_routing_net_set.get((u, v), [])])
                    ttl_criticality_v_u = sum([net.criticality for net in self.directed_tdm_edge_to_routing_net_set.get((v, u), [])])

                    if ttl_criticality_u_v > 0 or ttl_criticality_v_u > 0:
                        resource_u_v = data['resource_cnt'] * (ttl_criticality_u_v / (ttl_criticality_u_v + ttl_criticality_v_u))
                        resource_v_u = data['resource_cnt'] - resource_u_v
                        if ttl_criticality_u_v > 0:
                            self.refine(u, v, resource_u_v)
                        if ttl_criticality_v_u > 0:
                            self.refine(v, u, resource_v_u)

            """
            Static Update
            """
            delay = self.get_delay()

            prev_system_delay = self.system_delay
            self.system_delay = max(delay)
            self.update_criticality(delay)
            print(f'Delay: {self.system_delay}\tPrev Delay: {prev_system_delay}\t Abs Gap: '
                  f'{abs(round(self.system_delay - prev_system_delay, 4))}')

            if self.system_delay < self.min_system_delay:
                self.min_system_delay = self.system_delay
                self.best_dir_tdm_edge_path_tdm_ratio_matrix = copy.deepcopy(self.dir_tdm_edge_path_tdm_ratio_matrix)
                print(f'Minimum System Delay updated to {self.min_system_delay}')


            if abs(self.system_delay - prev_system_delay) < self.sig_thresh:
                break




if __name__ == '__main__':
    # HP Server
    # two_stage_reroute_res = "../../Res/Baseline/testcase6/run_20241117152129/run_20241117160605/two_stage_reroute_res.pkl"
    # Lenovo Server
    # two_stage_reroute_res = "../../Res/Baseline/testcase1/run_20250124193703/run_20250126095044/two_stage_reroute_res.pkl"
    # two_stage_reroute_res = "../../Res/Baseline/testcase7/run_20250124194145/run_20250126095514/two_stage_reroute_res.pkl"
    two_stage_reroute_res = "../../Res/Baseline/testcase10/run_20250124204806/run_20250126111448/two_stage_reroute_res.pkl"


    with open(two_stage_reroute_res, "rb") as f:
        nets, weighted_routing_resource_net, _ = pickle.load(f)

    init_assign_refine = Solver(nets, weighted_routing_resource_net, sig_thresh=0.1)
    init_assign_refine.perf_init_assign_and_refine()

