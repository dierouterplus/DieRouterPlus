import copy

import numpy as np


class SolverMixin:
    def __init__(self, nets, weighted_routing_resource_net, sig_thresh=1e-3, stag_rounds_thresh=10):
        """
        :param nets:
        :param weighted_routing_resource_net:
        Attributes of an edge:
          [1] resource_cnt : number of physical wires on the edge
          [2] type: 0/1 represents internal/external edge
          [3] weight: delay of the edge
          [4] pre_adjusted_weight: weight + 1/resource_cnt for TDM edge; weight for non-TDM edge.
          [5] usage: the number of nets using the edge
        :param sig_thresh: Compared to the shortest delay, an improvement is
        considered significant only when the gain exceeds sig_thresh.
        :param stag_rounds_thresh: If the number of rounds without significant improvement exceeds
        the stagnation rounds threshold, the iteration stops.
        """
        self.nets = nets
        self.weighted_routing_resource_net = weighted_routing_resource_net
        self.sig_thresh = sig_thresh
        self.stag_rounds_thresh = stag_rounds_thresh

        self.directed_tdm_edge_to_routing_net_set = SolverMixin.get_directed_tdm_edge_to_routing_net_set(self.nets,
                                                                                                         self.weighted_routing_resource_net)

        """
        Construct dict(directed tdm edge -> row_idx)
        """
        self.directed_tdm_edge_row_idx_map = self.get_directed_tdm_edge_row_idx_map()
        """
        Construct dict(net -> col_idx)
        The process also initialize the net.directed_tdm_edge_to_sink_idx_list field for each net in self.nets.
        directed_tdm_edge_to_sink_idx_list dict(e -> sink_idx)
        """
        self.net_to_base_col_idx_map = self.setup_col_idx_map()

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

        """
        (n_tdm_edge, n_ttl_path)
        """
        self.dir_tdm_edge_path_tdm_ratio_matrix = self.init_tdm_ratio_matrix()
        self.system_delay = max([net.criticality for net in self.nets])

        print('Original System Delay is: ', self.system_delay)

        """
        Keep the best result.
        """
        self.min_system_delay = self.system_delay
        self.best_dir_tdm_edge_path_tdm_ratio_matrix = copy.deepcopy(self.dir_tdm_edge_path_tdm_ratio_matrix)

    def init_tdm_ratio_matrix(self):
        """
        (n_tdm_edge, n_ttl_path)
        """

        dir_tdm_edge_path_tdm_ratio_matrix = np.zeros((len(self.directed_tdm_edge_row_idx_map),
                                                            self.non_tdm_delay_array.size))

        for directed_tdm_edge in self.directed_tdm_edge_to_routing_net_set:
            u, v = directed_tdm_edge
            tdm_ratio = self.weighted_routing_resource_net[u][v]['usage'] / self.weighted_routing_resource_net[u][v]['resource_cnt']
            routing_net_set = self.directed_tdm_edge_to_routing_net_set[directed_tdm_edge]
            row_idx = self.directed_tdm_edge_row_idx_map[directed_tdm_edge]
            for net in routing_net_set:
                base_col_idx = self.net_to_base_col_idx_map[net]
                for off_set in net.directed_tdm_edge_to_sink_idx_list[directed_tdm_edge]:
                    col_idx = base_col_idx + off_set
                    dir_tdm_edge_path_tdm_ratio_matrix[row_idx, col_idx] = tdm_ratio

        return dir_tdm_edge_path_tdm_ratio_matrix

    @staticmethod
    def get_directed_tdm_edge_to_routing_net_set(nets, weighted_routing_resource_net):
        """
        :return: dict((u, v) -> set)
        Map the directed TDM edge (u, v) to all nets routed through (u, v).
        """
        directed_tdm_edge_to_routing_net_set = {}
        for net in nets:
            for sol in net.routing_solutions:
                for u, v in zip(sol[0:], sol[1:]):
                    e = u, v
                    if weighted_routing_resource_net[u][v]['type'] == 1:
                        if e not in directed_tdm_edge_to_routing_net_set:
                            directed_tdm_edge_to_routing_net_set[e] = set()

                        """
                        Set is used to ensure that each net appears only once.
                        """
                        directed_tdm_edge_to_routing_net_set[e].add(net)

        return directed_tdm_edge_to_routing_net_set

    def get_directed_tdm_edge_row_idx_map(self):
        """
        return: dict(directed tdm edge -> row idx of tdm ratio matrix)
        """
        items = []
        row_idx = 0
        for u, v, data in self.weighted_routing_resource_net.edges(data=True):
            if data['type'] == 1:
                items.append(((u, v), row_idx))
                items.append(((v, u), row_idx+1))
                row_idx += 2
        return dict(items)


    def setup_col_idx_map(self):
        """
        return:
        The col_idx_map is used to map a (net, directed_tdm_edge) pair to a corresponding set of column indices.
        The mapping process involves first determining the base index associated with the net,
        then using the directed_tdm_edge to retrieve the offset relative to this base index.
        """

        net_to_base_col_idx_map = {}

        col_idx = 0
        for net in self.nets:
            net_to_base_col_idx_map[net] = col_idx

            net.directed_tdm_edge_to_sink_idx_list = {}

            for off_set, sol in enumerate(net.routing_solutions):
                for u, v in zip(sol[0:], sol[1:]):
                    if self.weighted_routing_resource_net[u][v]['type'] == 1:
                        e = (u, v)
                        if e not in net.directed_tdm_edge_to_sink_idx_list:
                            net.directed_tdm_edge_to_sink_idx_list[e] = []
                        net.directed_tdm_edge_to_sink_idx_list[e].append(off_set)

            col_idx += len(net.routing_solutions)
        return net_to_base_col_idx_map


    @staticmethod
    def get_non_tdm_edge_delay_array(nets, weighted_routing_resource_net):
        """
        return: An array where each element corresponds to the total delay on non-TDM type edges
        for the corresponding path.
        """
        non_tdm_path_delay_lst = []
        for net in nets:
            for sol in net.routing_solutions:
                non_tdm_path_delay = 0.0
                for u, v in zip(sol[0:], sol[1:]):
                    if weighted_routing_resource_net[u][v]['type'] == 0:
                        non_tdm_path_delay += 1

                non_tdm_path_delay_lst.append(non_tdm_path_delay)
        """
        (n_ttl_path, )
        """
        return np.array(non_tdm_path_delay_lst)


    @staticmethod
    def get_n_tdm_edge_array(nets, weighted_routing_resource_net):
        """
        :return: An array where each element represents the number of TDM edges on the path from src to sink.
        """
        n_tdm_edge_lst  = []
        for net in nets:
            for sol in net.routing_solutions:
                n_tdm_edge = 0
                for u, v in zip(sol[0:], sol[1:]):
                    if weighted_routing_resource_net[u][v]['type'] == 1:
                        n_tdm_edge += 1

                n_tdm_edge_lst.append(n_tdm_edge)
        """
        (n_ttl_path, )
        """
        return np.array(n_tdm_edge_lst)


    def get_delay(self):
        """
        :return:
        """
        tdm_delay = np.sum(self.dir_tdm_edge_path_tdm_ratio_matrix, axis=0) + 0.5 * self.n_tdm_edge_array
        delay = self.non_tdm_delay_array + tdm_delay
        return delay

    def update_criticality(self, delay, nets_to_update=None):
        """
        :return:
        """
        if nets_to_update is not None:
            for net in nets_to_update:
                base_col_idx = self.net_to_base_col_idx_map[net]
                for off_set in range(len(net.sink_criticality)):
                    net.sink_criticality[off_set] = delay[base_col_idx + off_set]

                net.criticality = max(net.sink_criticality)
        else:
            i = 0
            for net in self.nets:
                for j in range(len(net.sink_criticality)):
                    net.sink_criticality[j] = delay[i]
                    i += 1

                net.criticality = max(net.sink_criticality)



    def refine(self, u, v, resource_cnt):
        """
        For the directed edge (u, v), reallocate the TDM ratio for the nets utilizing this edge,
        given the resource count resource_cnt.
        :param u:
        :param v:
        :param resource_cnt:
        :return:
        """
        e = (u, v)
        row_idx = self.directed_tdm_edge_row_idx_map[e]

        """
        Extract tdm ratio for nets utilizing e.
        """
        net_tdm_ratio_map = {}
        for net in self.directed_tdm_edge_to_routing_net_set[e]:
            base_col_idx = self.net_to_base_col_idx_map[net]
            offset = net.directed_tdm_edge_to_sink_idx_list[e][0]
            col_idx = base_col_idx + offset
            matrix_idx = row_idx, col_idx
            """
            tdm_ratio of net on edge (u, v)
            """
            net_tdm_ratio_map[net] = self.dir_tdm_edge_path_tdm_ratio_matrix[matrix_idx]


        """
        Compute tdm_ratio_adjustment_factor
        """
        tdm_ratio_adjustment_factor = 0.0
        for net in net_tdm_ratio_map:
            tdm_ratio_adjustment_factor += net.criticality / net_tdm_ratio_map[net]

        tdm_ratio_adjustment_factor /= resource_cnt

        """
        Update dir_tdm_edge_path_tdm_ratio_matrix
        """
        for net in self.directed_tdm_edge_to_routing_net_set[e]:
            adjusted_tdm_ratio = tdm_ratio_adjustment_factor * net_tdm_ratio_map[net] / net.criticality
            base_col_idx = self.net_to_base_col_idx_map[net]
            for offset in net.directed_tdm_edge_to_sink_idx_list[e]:
                col_idx = base_col_idx + offset
                matrix_idx = row_idx, col_idx
                self.dir_tdm_edge_path_tdm_ratio_matrix[matrix_idx] = adjusted_tdm_ratio
