import numpy as np
from utils.timing import timeit

import multiprocessing as mp

class DPLegalMixIn:
    def gen_res_array(self, directed_tdm_edge, net_to_discrete_tdm_ratio):
        """
        The result array is computed and will be filled back into the dir_tdm_edge_path_tdm_ratio_matrix.
        :param directed_tdm_edge: (u, v)
        :param net_to_discrete_tdm_ratio:
        :return: array (n_ttl_path, )
        """

        if net_to_discrete_tdm_ratio is None:
            return None

        res_array = np.zeros(self.dir_tdm_edge_path_tdm_ratio_matrix.shape[1])
        for net, discrete_tdm_ratio in net_to_discrete_tdm_ratio.items():
            base_col_idx = self.net_to_base_col_idx_map[net]
            for off_set in net.directed_tdm_edge_to_sink_idx_list[directed_tdm_edge]:
                res_array[base_col_idx + off_set] = discrete_tdm_ratio

        return res_array


    def get_net_to_base_col_index_map(self):
        """
        :return: dict(net -> base col index)
        """
        net_to_base_col_idx_map = {}
        base_col_idx = 0
        for net in self.nets:
            net_to_base_col_idx_map[net] = base_col_idx
            base_col_idx += len(net.sinks)

        return net_to_base_col_idx_map

    def solve_tdm_edges_in_parallel(self, fpga_pair_task_id_tuples):
        """
        :param fpga_pair_task_id_tuples: [ ... (u, v, task_id) ... ]
        :return:
        """
        with mp.Pool(processes=self.n_processes) as pool:
            """
            res_tuples : list[ (row_idx_0, row_idx_1, res_array_0, res_array_1) ]
            """
            res_tuples = pool.map(self.solve_tdm_edge, fpga_pair_task_id_tuples)
        return res_tuples

    def solve_tdm_edges_in_sequence(self, fpga_pair_task_id_tuples):
        """
        :param fpga_pair_task_id_tuples: [ ... (u, v, task_id) ... ]
        :return:
        """

        """
        res_tuples : list[ (row_idx_0, row_idx_1, res_array_0, res_array_1) ]
        """
        res_tuples = []
        for fpga_pair_task_id_tuple in fpga_pair_task_id_tuples:
            res_tuples.append(self.solve_tdm_edge(fpga_pair_task_id_tuple))

        return res_tuples


    def get_system_delay(self):
        """
        :return:
        """
        tdm_delay = np.sum(self.dir_tdm_edge_path_tdm_ratio_matrix, axis=0) + 0.5 * self.n_tdm_edge_array
        delay = self.non_tdm_delay_array + tdm_delay
        return np.max(delay)

    @timeit("perform_dp_legal")
    def perform_dp_legal(self):

        fpga_pair_task_id_tuples = []
        task_id = 0
        for u, v, data in self.weighted_routing_resource_net.edges(data=True):
            if data['type'] == 1:
                if (u, v) in self.directed_tdm_edge_to_routing_net_set or (v, u) in self.directed_tdm_edge_to_routing_net_set:
                    fpga_pair_task_id_tuples.append((u, v, task_id))
                    task_id += 1

        print('Beginning Solving Dynamical Programming')
        """
        res_tuples : list[ (row_idx_0, row_idx_1, res_array_0, res_array_1) ]
        """
        if self.enable_multiprocessing:
            res_tuples = self.solve_tdm_edges_in_parallel(fpga_pair_task_id_tuples)
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