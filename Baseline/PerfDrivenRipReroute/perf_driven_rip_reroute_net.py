import pickle
import heapq
import time
from copy import deepcopy

import math

from Baseline.VioRipReroute.vio_rip_reroute import VioRipReroute
from Baseline.baseline_utils.metrics import get_net_criticality
from Baseline.Net.net import Net
from functools import partial
import networkx as nx
from tqdm import tqdm

class PerfDrivenRipRerouteNet:
    def __init__(self,
                 nets,
                 weighted_routing_resource_net,
                 edge_to_routing_net_set,
                 edge_criticality_metric='Max',
                 patience=10):
        """
        :param nets:
        :param weighted_routing_resource_net:
        Attributes of an edge:
          [1] resource_cnt : number of physical wires on the edge
          [2] type: 0/1 represents internal/external edge
          [3] weight: delay of the edge
          [4] pre_adjusted_weight: weight + 1/resource_cnt for TDM edge; weight for non-TDM edge.
          [5] usage: the number of nets using the edge
        :param edge_to_routing_net_set
            dict(edge -> set(nets))
        :param edge_criticality_metric: 'Max' or 'Sum' of net criticality
        :param patience: If the result remains the same for patience consecutive iterations, terminate the process.
        """
        self.nets = nets
        self.weighted_routing_resource_net = weighted_routing_resource_net
        self.edge_to_routing_net_set = edge_to_routing_net_set

        self.edge_criticality_metric = edge_criticality_metric

        self.system_delay = max([get_net_criticality(self.weighted_routing_resource_net, net) for net in self.nets])
        print('System delay after eliminating the violations is {}'.format(self.system_delay))

        """
        Store the optimal results obtained during the iteration process.
        """
        self.opt_system_delay = math.inf
        self.opt_nets = None
        self.opt_weighted_routing_resource_net = None

        self.patience = patience




    def get_edge_criticality(self, u, v):
        """
        edge_criticality is the sum of the criticality of all nets passing through it.
        :param u:
        :param v:
        :return:
        """
        e = min(u, v), max(u, v)

        assert self.edge_criticality_metric == 'Sum' or self.edge_criticality_metric == 'Max'
        if self.edge_criticality_metric == 'Sum':
            edge_criticality = 0.0
            for net in self.edge_to_routing_net_set[e]:
                edge_criticality += get_net_criticality(self.weighted_routing_resource_net, net)
        elif self.edge_criticality_metric == 'Max':
            edge_criticality = max([get_net_criticality(self.weighted_routing_resource_net, net)
                                    for net in self.edge_to_routing_net_set[e]])
        return edge_criticality

    def extract_most_critical_edge(self):
        maximal_criticality = -1 * math.inf
        most_critical_edge = None
        for u, v, data in self.weighted_routing_resource_net.edges(data=True):
            if data['type'] == 1:
                criticality = self.get_edge_criticality(u, v)
                if criticality > maximal_criticality:
                    maximal_criticality = criticality
                    most_critical_edge = min(u, v), max(u, v)

        return most_critical_edge


    def rip_up(self, net: Net):
        """
        :param net:
        :return:
        """

        """
        Remove duplicated edge in net.routing_solutions.
        """
        edges_to_rip_up = set()
        for sol in net.routing_solutions:
            for u, v in zip(sol[0:], sol[1:]):
                e = min(u, v), max(u, v)
                edges_to_rip_up.add(e)

        for u, v in edges_to_rip_up:
            """
            Step 1. Adjust usages of all related edges. 
                    Adjust weights of all related external edges.
            """
            self.weighted_routing_resource_net[u][v]['usage'] -= 1
            if self.weighted_routing_resource_net[u][v]['type'] == 1:
                self.weighted_routing_resource_net[u][v]['weight'] -= 1 / self.weighted_routing_resource_net[u][v]['resource_cnt']
                self.weighted_routing_resource_net[u][v]['pre_adjusted_weight'] -= 1 / self.weighted_routing_resource_net[u][v][
                    'resource_cnt']

            """
            Step 2. Del net from edge_to_routing_net_set[edge].
            """
            e = min(u, v), max(u, v)
            self.edge_to_routing_net_set[e].remove(net)

        net.criticality = None
        net.sink_criticality = None
        net.routing_solutions = None



    @staticmethod
    def get_dijkstra_weight(u, v, data):
        if data['type'] == 0:
            if data['usage'] < data['resource_cnt']:
                return 1.0
            else:
                return 1e9
        else:
            return data['pre_adjusted_weight']


    def reroute(self, net: Net):
        _, path_dict = nx.single_source_dijkstra(self.weighted_routing_resource_net,
                                                 net.src,
                                                 weight=PerfDrivenRipRerouteNet.get_dijkstra_weight)

        """
        Step 1. Reroute
        Note that net.criticality and net.sink_criticality will be updated when get_net_criticality is called.
        """
        assert net.routing_solutions is None
        net.routing_solutions = []
        for sink in net.sinks:
            net.routing_solutions.append(path_dict[sink])

        """
        Remove duplicated edge in net.routing_solutions.
        """
        rerouting_edges = set()
        for sol in net.routing_solutions:
            for u, v in zip(sol[0:], sol[1:]):
                e = min(u, v), max(u, v)
                rerouting_edges.add(e)

        for u, v in rerouting_edges:
            """
            Step 2. Update edge_to_routing_net_id_set
            """
            e = min(u, v), max(u, v)
            self.edge_to_routing_net_set[e].add(net)

            """
            Step 3. Update weighted_routing_resource_network
            """
            self.weighted_routing_resource_net[u][v]['usage'] += 1
            if self.weighted_routing_resource_net[u][v]['type'] == 1:
                self.weighted_routing_resource_net[u][v]['weight'] += 1 / self.weighted_routing_resource_net[u][v]['resource_cnt']
                self.weighted_routing_resource_net[u][v]['pre_adjusted_weight'] += 1 / self.weighted_routing_resource_net[u][v]['resource_cnt']

    def perf_driven_rip_reroute(self):

        prev_system_delay = None
        repeated_system_delay_cnt = 0

        while True:
            e = self.extract_most_critical_edge()

            """
            Calculate the nets that need to be removed, and sort them in descending order based on criticality.
            """
            nets_to_rip_up = list(self.edge_to_routing_net_set[e])
            nets_to_rip_up.sort(key=lambda net: get_net_criticality(self.weighted_routing_resource_net, net),
                                reverse=True)

            for net in nets_to_rip_up:
                """
                sink_idx is an index in net.sinks. 
                Here, the indices are sorted based on the sinks' criticality.
                """
                self.rip_up(net)
                self.reroute(net)


            """
            Recalculate criticality.
            """
            for net in self.nets:
                net.criticality = None
            self.system_delay = max(
                [get_net_criticality(self.weighted_routing_resource_net, net) for net in self.nets])

            """
            Record the best result.
            """
            if self.system_delay < self.opt_system_delay:
                self.opt_system_delay = self.system_delay
                self.opt_nets = deepcopy(self.nets)
                self.opt_weighted_routing_resource_net = deepcopy(self.weighted_routing_resource_net)

            print('System delay is {}'.format(self.system_delay))


            """
            Check the termination condition. If the results remain the same for patience 
            consecutive iterations, terminate the process.
            """
            if self.system_delay == prev_system_delay:
                repeated_system_delay_cnt += 1
            else:
                repeated_system_delay_cnt = 0
                prev_system_delay = self.system_delay

            if repeated_system_delay_cnt >= self.patience:
                break

        return self.opt_nets, self.opt_weighted_routing_resource_net, self.opt_system_delay



if __name__ == '__main__':
    # hybrid_init_routing_res_path = "/home/huqf/FPGADieRouting/Res/Baseline/testcase4/run_20241113111933/hyb_init_routing_res.pkl"
    # hybrid_init_routing_res_path = "/home/huqf/FPGADieRouting/Res/Baseline/testcase5/run_20241113111752/hyb_init_routing_res.pkl"
    # hybrid_init_routing_res_path = "/home/huqf/FPGADieRouting/Res/Baseline/testcase6/run_20241111193533/hyb_init_routing_res.pkl"
    # hybrid_init_routing_res_path = "/home/huqf/FPGADieRouting/Res/Baseline/testcase7/run_20241112201514/hyb_init_routing_res.pkl"
    # hybrid_init_routing_res_path = "/home/huqf/FPGADieRouting/Res/Baseline/testcase8/run_20241113093433/hyb_init_routing_res.pkl"
    hybrid_init_routing_res_path = "/home/huqf/FPGADieRouting/Res/Baseline/testcase9/run_20241113112804/hyb_init_routing_res.pkl"
    # hybrid_init_routing_res_path = "/home/huqf/FPGADieRouting/Res/Baseline/testcase10/run_20241113114751/hyb_init_routing_res.pkl"

    with open(hybrid_init_routing_res_path, "rb") as f:
        nets, weighted_routing_resource_network = pickle.load(f)

    start = time.time()
    vio_rip_rerouter = VioRipReroute(nets, weighted_routing_resource_network)
    vio_rip_rerouter.rip_up_and_reroute()
    end = time.time()
    print('Violation Rip Up and Reroute took {} seconds'.format(end - start))

    start = time.time()
    perf_driven_reroute = PerfDrivenRipRerouteNet(vio_rip_rerouter.nets,
                                                  vio_rip_rerouter.weighted_routing_resource_net,
                                                  vio_rip_rerouter.edge_to_routing_net_set)

    opt_nets, opt_weighted_routing_resource_net, opt_system_delay = perf_driven_reroute.perf_driven_rip_reroute()
    end = time.time()
    print('Performance-Driven Rip Up and Reroute took {} seconds'.format(end - start))
    print('Opt system delay is {}'.format(opt_system_delay))

    """
    Check Edge Usage
    """
    edge_usage = {}
    for net in opt_nets:
        edge_set = set()
        """
        Ensure that the path contains no cycles.
        """
        for sol in net.routing_solutions:
            for u, v in zip(sol[0:], sol[1:]):
               e = u, v
               edge_set.add(e)

        for u, v in edge_set:
             e = min(u, v), max(u, v)
             if e not in edge_usage:
                 edge_usage[e] = 0
             edge_usage[e] += 1

    """
    Calculate the delay of tdm type edge.
    """
    edge_delay = {}
    for u, v, data in opt_weighted_routing_resource_net.edges(data=True):
        e = min(u, v), max(u, v)
        assert edge_usage[e] == data['usage']
        if data['type'] == 1:
            edge_delay[e] = data['usage'] / data['resource_cnt']

    """
    Check Delay.
    """
    max_sink_delay = -1 * math.inf
    avg_err = 0.0
    err_cnt = 0
    for net in opt_nets:
        for idx, sol in enumerate(net.routing_solutions):
            sink_delay = 0.0
            for u, v in zip(sol[0:], sol[1:]):
                e = min(u, v), max(u, v)
                if e not in edge_delay:
                    sink_delay += 1.0
                else:
                    sink_delay += (0.5 + edge_delay[e])

            avg_err += abs(sink_delay - net.sink_criticality[idx])
            err_cnt += 1

            if sink_delay > max_sink_delay:
                max_sink_delay = sink_delay

    print(f'average error:{avg_err / err_cnt}')
    print(abs(max_sink_delay - opt_system_delay))







