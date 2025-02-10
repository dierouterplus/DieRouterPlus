import networkx as nx
from Baseline.Net.net import Net
from collections import deque

class HybridInitialRouting:
    def __init__(self,
                 nets,
                 weighted_routing_resource_network: nx.Graph,
                 routing_config=None,
                 n_pins_gap_factor=60):
        """
        :param nets: a list of Net objects.
        :param weighted_routing_resource_network:
             The attributes of the edge include:
             resource_cnt : number of physical wires on the edge
             type: 0/1 internal/external edge
             weight: delay of the edge
             - weight of tdm type edge is initialized to 0.5.
             - weight of non-tdm type edge is initialized to 1.0.
             usage: the number of nets using the edge
        :param routing_config: dict('critical_net' -> 'SMT' or 'Dijkstra' or 'MST'
        'non_critical_net' -> 'SMT' or 'Dijkstra' or 'MST')
        :param n_pins_gap_factor: parameter to determine the critical nets and non-critical nets.
        """
        self.nets = nets
        self.weighted_routing_resource_network = weighted_routing_resource_network
        self.routing_config = routing_config

        if self.routing_config is None:
            self.routing_config = {'critical_net': 'SMT', 'non_critical_net': 'Dijkstra'}

        self.routing_func_dict = {
            'SMT': self.perform_minimum_steiner_tree_routing,
            'Dijkstra': self.perform_dijkstra_routing,
            'MST': self.perform_minimum_spanning_tree_routing,
        }

        self.add_pre_adjusted_weight_field()

        self.n_pins_gap_factor = n_pins_gap_factor
        self.is_sorted_by_n_pins = False
        self.is_sorted_by_n_sinks = False




    def add_pre_adjusted_weight_field(self):
        """
        When a net passes through a TDM-type edge, the weight of the TDM-type edge will increase.
        Therefore, the weight of TDM-type edges is pre-adjusted here to account for this effect.
        :return:
        """
        for u, v, data in self.weighted_routing_resource_network.edges(data=True):
            if data['type'] == 1:
                data['pre_adjusted_weight'] = data['weight'] + 1.0 / data['resource_cnt']
            else:
                data['pre_adjusted_weight'] = data['weight']






    def sort_nets_by_n_pins(self):
        assert self.is_sorted_by_n_sinks is False
        if not self.is_sorted_by_n_pins:
            self.nets.sort(key=lambda x: x.n_pins, reverse=True)
            self.is_sorted_by_n_pins = True


    def sort_nets_by_n_sinks(self):
        assert self.is_sorted_by_n_pins is False
        if not self.is_sorted_by_n_sinks:
            self.nets.sort(key=lambda x: len(x.sinks), reverse=True)
            self.is_sorted_by_n_sinks = True


    def split_critical_sets(self):
        self.sort_nets_by_n_pins()
        n_pins_lst = [net.n_pins for net in self.nets]
        sum_prefix  = 0
        sum_suffix = sum(n_pins_lst)

        for i in range(0, len(n_pins_lst)):
            sum_prefix += n_pins_lst[i]
            sum_suffix -= n_pins_lst[i]
            if 0 < sum_suffix <= sum_prefix / self.n_pins_gap_factor:
                """
                Critical Set        : 0 ~ i+1
                Non-Critical Set    : i+1 ~ len(n_pins_lst)
                """
                return list(range(0, i+1)), list(range(i+1,len(n_pins_lst)))

        """
        Critical set is empty.
        """
        return list(), list(range(len(n_pins_lst)))

    def perform_minimum_steiner_tree_routing(self, net:Net):
        T = nx.approximation.steiner_tree(self.weighted_routing_resource_network,
                                          [net.src] + net.sinks,
                                          weight='pre_adjusted_weight')

        """
        Perform a breadth-first traversal of the approximated minimum steiner tree to construct the routing results. 
        """
        nodes_to_vst = deque()
        nodes_to_vst.append(net.src)
        path_dict = {net.src: [net.src]}
        visited = {net.src}

        while len(nodes_to_vst) > 0:
            u = nodes_to_vst.popleft()
            path_to_u = path_dict[u]
            for v in T.neighbors(u):
                if v not in visited:
                    visited.add(v)
                    path_dict[v] = path_to_u + [v]
                    nodes_to_vst.append(v)

        """
        Store the path.
        """
        net.routing_solutions = [path_dict[sink] for sink in net.sinks]

    def perform_dijkstra_routing(self, net:Net):
        """
        :param net:
        :return:
        """

        """
        (dist_dict, path_dict)
        """
        single_src_dist_path_dict_tuple = nx.single_source_dijkstra(self.weighted_routing_resource_network,
                                                                    net.src,
                                                                    weight='pre_adjusted_weight')


        net.routing_solutions = []
        for dst in net.sinks:
            net.routing_solutions.append(single_src_dist_path_dict_tuple[1][dst])



    def perform_minimum_spanning_tree_routing(self, net:Net):
        """
        :param net:
        :return:
        """
        T = nx.minimum_spanning_tree(self.weighted_routing_resource_network,
                                     weight='pre_adjusted_weight')

        single_src_dist_path_dict_tuple = nx.single_source_dijkstra(T,
                                                                    net.src,
                                                                    weight='pre_adjusted_weight')

        net.routing_solutions = []
        for dst in net.sinks:
            net.routing_solutions.append(single_src_dist_path_dict_tuple[1][dst])





    def update_edge_info(self, net):
        """
        Use routing solutions of the net to update edge info: edge usage and edge weight.
        :param net:
        :return:
        """

        """
        The same edge may appear in paths from src leading to different pins. 
        However, for a net, each using edge should only be counted once.
        """
        routing_edge_set = set()
        for sol in net.routing_solutions:
            for u, v in zip(sol[0:], sol[1:]):
                routing_edge_set.add((u, v))

        for u, v in routing_edge_set:
            self.weighted_routing_resource_network[u][v]['usage'] += 1
            if self.weighted_routing_resource_network[u][v]['type'] == 1:
                resource_cnt = self.weighted_routing_resource_network[u][v]['resource_cnt']
                self.weighted_routing_resource_network[u][v]['weight'] += 1 / resource_cnt
                self.weighted_routing_resource_network[u][v]['pre_adjusted_weight'] += 1 / resource_cnt

    def perform_routing(self):
        """
        :return:
        nets, weighted_routing_resource_network

        The attributes of the edge include:
        resource_cnt : number of physical wires on the edge
        type: 0/1 internal/external edge
        weight: delay of the edge
        usage: the number of nets using the edge
        """

        # Sort all nets in non-decreasing order of their pin numbers
        self.sort_nets_by_n_pins()

        """
        critical_set: List. Index in the sorted nets
        non_critical_set: List. Index in the sorted nets
        """
        critical_set, non_critical_set = self.split_critical_sets()


        """
        Deal with critical set
        """
        for i in critical_set:
            self.routing_func_dict[self.routing_config['critical_net']](self.nets[i])
            self.update_edge_info(self.nets[i])

        """
        Deal with non-critical set
        """
        for i in non_critical_set:
            """
            single_src_dist_path_dict_tuple : (dist_dict, path_dict)
            """
            self.routing_func_dict[self.routing_config['non_critical_net']](self.nets[i])
            self.update_edge_info(self.nets[i])

        return self.nets, self.weighted_routing_resource_network



if __name__ == '__main__':
    from Baseline.baseline_utils.process_raw_data import build_weighted_routing_resource_network_and_nets
    import time
    testcase_dir = '../../Data/testcase5'

    weighted_routing_resource_network, nets = build_weighted_routing_resource_network_and_nets(testcase_dir)
    hybrid_init_routing = HybridInitialRouting(nets, weighted_routing_resource_network,
                                               routing_config={'critical_net': 'Dijkstra', 'non_critical_net': 'Dijkstra'}
                                               # routing_config={'critical_net': 'Dijkstra',
                                               #                 'non_critical_net': 'SMT'}
                                               )


    start_time = time.time()
    nets, weighted_routing_resource_network = hybrid_init_routing.perform_routing()
    end_time = time.time()
    print(end_time - start_time)

    import pickle
    res_path = r'./res_to_comp_debug_1.pkl'
    with open(res_path, 'rb') as f:
        load_nets, load_weighted_routing_resource_network = pickle.load(f)

    for net, load_net in zip(nets, load_nets):
        for sol, load_sol in zip(net.routing_solutions, load_net.routing_solutions):
            assert sol == load_sol

    for u, v in weighted_routing_resource_network.edges():
        assert weighted_routing_resource_network[u][v] == load_weighted_routing_resource_network[u][v]

