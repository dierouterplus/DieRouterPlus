import pickle
import numpy as np
import sys
sys.path.append('../../')
from utils.timing import timeit



import multiprocessing as mp
from Baseline.Discretization.incremental_dp_legal import IncrementalDPLegal

class BalancedIncrementalDPLegal(IncrementalDPLegal):
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
        super(BalancedIncrementalDPLegal, self).__init__(nets,
                                                         weighted_routing_resource_net,
                                                         dir_tdm_edge_path_tdm_ratio_matrix,
                                                         directed_tdm_edge_row_idx_map,
                                                         enable_multiprocessing,
                                                         n_processes)


    def solve_tdm_edges(self, tasks):
        """
        :param tasks: list(tuple(u, v, task_id, workload))
        :return: list(tuple(row_idx_0, row_idx_1, res_array_0, res_array_1))
        """

        res_tuple_list = []
        for task in tasks:
            res_tuple_list.append(self.solve_tdm_edge(task[0:-1]))
            print(f"Task-{task[2]} has been solved")

        return res_tuple_list






    def solve_tdm_edges_in_parallel(self, tasks):
        """
        :param tasks: List( List( tuple(u, v, task_id, workload) ) )
        :return:
        """
        with mp.Pool(processes=self.n_processes) as pool:
            """
            res_tuples : list[ tuple(row_idx_0, row_idx_1, res_array_0, res_array_1) ]
            """
            results = []
            for task in tasks:
                result = pool.apply_async(self.solve_tdm_edges, args=(task, ))
                results.append(result)

            res_tuples = []
            for result in results:
                res_tuples.extend(result.get())

        return res_tuples



    def get_workload(self, u, v):
        """
        :param u:
        :param v:
        :return:
        n_nets[0], n_nets[1]: Number of nets traversing (u, v) and (v, u)
        n_wires[0], n_wires[1]: The approximated resource usage on (u, v) and (v, u)
        """
        n_nets, n_wires = [], []
        for e in [(u, v), (v, u)]:
            n_nets.append(len(self.directed_tdm_edge_to_routing_net_set.get(e, [])))
            row_idx = self.directed_tdm_edge_to_row_idx[e]
            n_wire = 0
            if e in self.directed_tdm_edge_to_routing_net_set:
                for net in self.directed_tdm_edge_to_routing_net_set[e]:
                    base_col_idx = self.net_to_base_col_idx_map[net]
                    """
                    A net may have multiple pins, but not all of them pass through edge e. 
                    Therefore, it is necessary to identify the first pin corresponding to edge e.
                    """
                    offset = net.directed_tdm_edge_to_sink_idx_list[e][0]
                    conti_tdm_ratio = self.dir_tdm_edge_path_tdm_ratio_matrix[row_idx, base_col_idx+offset]
                    n_wire += 1.0/conti_tdm_ratio

            n_wires.append(n_wire)

        return n_nets[0] * n_wires[0] + n_nets[1] * n_wires[1]




    def split_tasks(self, fpga_pair_task_id_tuples):
        """
        :param fpga_pair_task_id_tuples: list( tuple(u, v, task_id, workload) )
        :return: list( list( tuple(u, v, task_id, workload) ) )
        """

        fpga_pair_task_id_tuples.sort(key=lambda x: x[3], reverse=True)

        if len(fpga_pair_task_id_tuples) < self.n_processes:
            self.n_processes = len(fpga_pair_task_id_tuples)

        usage_array = np.zeros(self.n_processes)
        res = [[] for _ in range(self.n_processes)]

        for fpga_pair_task_id_tuple in fpga_pair_task_id_tuples:
            min_load_cpu = np.argmin(usage_array)
            res[min_load_cpu].append(fpga_pair_task_id_tuple)
            usage_array[min_load_cpu] += fpga_pair_task_id_tuple[3]

        return res






    @timeit("perform_dp_legal")
    def perform_dp_legal(self):
        """
        (u, v, task_id, workload)
        """
        fpga_pair_task_id_tuples = []
        task_id = 0
        for u, v, data in self.weighted_routing_resource_net.edges(data=True):
            if data['type'] == 1:
                if (u, v) in self.directed_tdm_edge_to_routing_net_set or (v, u) in self.directed_tdm_edge_to_routing_net_set:
                    workload = self.get_workload(u, v)
                    fpga_pair_task_id_tuples.append((u, v, task_id, workload))
                    task_id += 1


        """
        list( list( tuple(u, v, task_id, workload) ) )
        """
        tasks = self.split_tasks(fpga_pair_task_id_tuples)

        print('Beginning Solving Dynamical Programming with Load Balance')
        """
        res_tuples : list[ (row_idx_0, row_idx_1, res_array_0, res_array_1) ]
        """
        if self.enable_multiprocessing:
            res_tuples = self.solve_tdm_edges_in_parallel(tasks)
        else:
            res_tuples = self.solve_tdm_edges_in_sequence(fpga_pair_task_id_tuples)
        print('Finished Solving Dynamical Programming')

        print('Beginning Setting Legalized Solutions')

        for res_tuple in res_tuples:
            res_lst = [(res_tuple[0], res_tuple[2]), (res_tuple[1], res_tuple[3])]

            for res in res_lst:
                row_idx, res_array = res
                if res_array is not None:
                    self.dir_tdm_edge_path_tdm_ratio_matrix[row_idx] = res_array
        print('Finished Setting Legalized Solutions')
        system_delay = self.get_system_delay()
        print(f'After legalization, System delay is {system_delay}.')




if __name__ == '__main__':
    # HP Server
    # conti_sol_res = "/home/huqifu/FPGADieRouting/Res/Baseline/testcase4/run_20250105114424/run_20250105114432/run_20250105114445/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqifu/FPGADieRouting/Res/Baseline/testcase4/run_20250105114424/run_20250105121111/run_20250105121133/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqifu/FPGADieRouting/Res/Baseline/testcase5/run_20250105114536/run_20250105114542/run_20250105114552/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqifu/FPGADieRouting/Res/Baseline/testcase5/run_20250105114536/run_20250105121229/run_20250105121239/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqifu/FPGADieRouting/Res/Baseline/testcase6/run_20250105113012/run_20250105113758/run_20250105114000/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqifu/FPGADieRouting/Res/Baseline/testcase6/run_20250105113012/run_20250105121359/run_20250105121457/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqifu/FPGADieRouting/Res/Baseline/testcase7/run_20250105114709/run_20250105114750/run_20250105114830/conti_tdm_ratio.pkl"

    # conti_sol_res = "/home/huqifu/FPGADieRouting/Res/Baseline/testcase7/run_20250105114709/run_20250105121700/run_20250105121726/conti_tdm_ratio.pkl"

    # conti_sol_res = "/home/huqifu/FPGADieRouting/Res/Baseline/testcase8/run_20250105115411/run_20250105115447/run_20250105115526/conti_tdm_ratio.pkl"

    # conti_sol_res = "/home/huqifu/FPGADieRouting/Res/Baseline/testcase8/run_20250105115411/run_20250105121910/run_20250105121941/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqifu/FPGADieRouting/Res/Baseline/testcase8/run_20250105210926/run_20250105211147/run_20250105211255/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqifu/FPGADieRouting/Res/Baseline/testcase8/run_20250105211828/run_20250105212043/run_20250105212144/conti_tdm_ratio.pkl"

    # conti_sol_res = "/home/huqifu/FPGADieRouting/Res/Baseline/testcase8/run_20250105212911/run_20250105213637/run_20250105213725/conti_tdm_ratio.pkl"


    # Lenovo Server
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase1/run_20250106153257/run_20250106153306/run_20250106153339/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase1/run_20250106142650/run_20250106145032/run_20250106145814/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase1/run_20250106142650/run_20250106142708/run_20250106142733/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase2/run_20250106140550/run_20250106140602/run_20250106140621/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase2/run_20250106135828/run_20250106141318/run_20250106141559/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase2/run_20250106135828/run_20250106135841/run_20250106140316/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase2/run_20250106135828/run_20250106135841/run_20250106135852/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase2/run_20250106152810/run_20250106152823/run_20250106152846/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase3/run_20250106134918/run_20250106135008/run_20250106135114/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase4/run_20241113111933/run_20241120170743/run_20241126195408/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase4/run_20241113111933/run_20241231152559/run_20241231152611/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase4/run_20241231152840/run_20241231152854/run_20241231152913/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase5/run_20241113111752/run_20241120170927/run_20241126200029/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase6/run_20241111193533/run_20241119093605/run_20241126174313/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase6/run_20250106153542/run_20250106160053/run_20250106160236/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase7/run_20241112201514/run_20241119141119/run_20241125164618/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase8/run_20241113093433/run_20241119165208/run_20241126170653/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase8/run_20250106144824/run_20250106145147/run_20250106152339/conti_tdm_ratio.pkl"


    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase9/run_20241113112804/run_20241119170429/run_20241126182817/conti_tdm_ratio.pkl"
    conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase9/run_20250106172902/run_20250106175028/run_20250106181620/conti_tdm_ratio.pkl"

    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase10/run_20241113114751/run_20241120144955/run_20241126204916/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase9/run_20241230100803/run_20241230103031/run_20241230104250/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase9/run_20241230155746/run_20241230161013/run_20241230164811/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase9/run_20241113112804/run_20241230194456/run_20241230200034/conti_tdm_ratio.pkl"

    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase10/run_20241230101911/run_20241230110612/run_20241230134328/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase10/run_20241230160523/run_20241230170403/run_20241230181116/conti_tdm_ratio.pkl"
    # conti_sol_res = "/home/huqf/FPGADieRouting/Res/Baseline/testcase10/run_20241113114751/run_20241231093945/run_20241231120333/conti_tdm_ratio.pkl"

    with open(conti_sol_res, "rb") as f:
        nets, weighted_routing_resource_net, dir_tdm_edge_path_tdm_ratio_matrix, directed_tdm_edge_row_idx_map = pickle.load(f)

    dp_legalizer = BalancedIncrementalDPLegal(nets, weighted_routing_resource_net, dir_tdm_edge_path_tdm_ratio_matrix, directed_tdm_edge_row_idx_map,
                                              n_processes=16)

    dp_legalizer.perform_dp_legal()














