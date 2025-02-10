import time

import math
import pickle
from copy import deepcopy

from Source.Baseline.VioRipReroute.vio_rip_reroute import VioRipReroute
from Source.Baseline.baseline_utils.metrics import get_net_criticality
from Source.Baseline.Net.net import Net
from functools import partial
import networkx as nx


class PerfDrivenRipRerouteSink:
    def __init__(self,
                 nets,
                 weighted_routing_resource_net,
                 edge_to_routing_net_set,
                 edge_criticality_metric='Max',
                 occupied_edge_cost='zero',
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
        :param occupied_edge_cost:
            'zero':  the cost of occupied edge is set to zero
            'weight': the cost of occupied edge is set to its weight
        :param patience: If the result remains the same for patience consecutive iterations, terminate the process.
        """
        self.nets = nets
        self.weighted_routing_resource_net = weighted_routing_resource_net
        self.edge_to_routing_net_set = edge_to_routing_net_set

        self.edge_criticality_metric = edge_criticality_metric
        self.occupied_edge_cost = occupied_edge_cost

        """
        For each net, set the mapping from edge to sink_idx_set.
        sink_idx in sink_idx_set iff edge lies on the routing path of from src to sol[sink_idx].
        """
        self.setup_net_edge_sink_mappings()

        self.system_delay = max([get_net_criticality(self.weighted_routing_resource_net, net) for net in self.nets])
        print('System delay after eliminating the violations is {}'.format(self.system_delay))

        """
        Store the optimal results obtained during the iteration process.
        """
        self.opt_system_delay = self.system_delay
        self.opt_nets = deepcopy(self.nets)
        self.opt_weighted_routing_resource_net = deepcopy(self.weighted_routing_resource_net)

        self.patience = patience


    def setup_net_edge_sink_mappings(self):
        """
        For each net, there is a dict((u, v) -> set(sink_idx)).
        :return:
        """
        for net in self.nets:
            net.edge_to_sink_idx_set = {}
            for sink_idx, sol in enumerate(net.routing_solutions):
                for u, v in zip(sol[0:], sol[1:]):
                    e = min(u, v), max(u, v)
                    if e not in net.edge_to_sink_idx_set:
                        net.edge_to_sink_idx_set[e] = set()
                    net.edge_to_sink_idx_set[e].add(sink_idx)


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


    def rip_up_sink(self, net: Net, sink_idx):
        """
        Remove the routing path in net corresponding to sink_idx.
        :param net:
        :param sink_idx:
        :return:
        """
        ripped_up_sol = net.routing_solutions[sink_idx]

        """
        Rip-up net.
        """
        net.routing_solutions[sink_idx] = []
        net.criticality = None

        """
        Update net.edge_to_sink_idx_set and edge_to_routing_net_set
        If edge_to_sink_idx_set[e] is an empty set, it indicates that the net no longer uses edge e.
        Then, we need to adjust statistics on edge e.
        """
        for u, v in zip(ripped_up_sol[0:], ripped_up_sol[1:]):
            e = min(u, v), max(u, v)
            net.edge_to_sink_idx_set[e].remove(sink_idx)

            if len(net.edge_to_sink_idx_set[e]) == 0:
                del net.edge_to_sink_idx_set[e]
                self.edge_to_routing_net_set[e].remove(net)

                data = self.weighted_routing_resource_net[u][v]
                if data['type'] == 1:
                    data['weight'] -= 1.0 / data['resource_cnt']
                    data['pre_adjusted_weight'] -= 1.0 / data['resource_cnt']

                """
                Bug Fix
                """
                data['usage'] -= 1

        return ripped_up_sol


    @staticmethod
    def get_dijkstra_weight(u, v, data, net: Net, occupied_edge_cost):
        edge = min(u, v), max(u, v)
        if edge in net.edge_to_sink_idx_set:
            assert occupied_edge_cost == 'zero' or occupied_edge_cost == 'weight'
            if occupied_edge_cost == 'zero':
                return 0.0
            else:
                return data['weight']
        elif data['type'] == 0:
            if data['usage'] < data['resource_cnt']:
                return 1.0
            else:
                return 1e9
        else:
            return data['pre_adjusted_weight']


    def reroute_sink(self, net: Net, sink_idx, ripped_up_sol):
        weight_func = partial(PerfDrivenRipRerouteSink.get_dijkstra_weight,
                              net=net,
                              occupied_edge_cost=self.occupied_edge_cost)

        rerouted_sol = nx.dijkstra_path(self.weighted_routing_resource_net,
                                        source=net.src,
                                        target=net.sinks[sink_idx],
                                        weight=weight_func)

        """
        Compute delay of the rerouted solution.
        If e has been used by other sinks, the delay of e is characterized by weight.
        Otherwise, the delay of e is characterized by pre_adjusted_weight.
        """
        reroute_sol_delay = 0.0
        for u, v in zip(rerouted_sol[0:], rerouted_sol[1:]):
            e = min(u, v), max(u, v)
            if e in net.edge_to_sink_idx_set:
                reroute_sol_delay += self.weighted_routing_resource_net[u][v]['weight']
            else:
                reroute_sol_delay += self.weighted_routing_resource_net[u][v]['pre_adjusted_weight']


        """
        Accept or Reject the rerouted solution.
        """
        if self.system_delay <= reroute_sol_delay:
            rerouted_sol = ripped_up_sol

        """
        Reroute net
        """
        net.routing_solutions[sink_idx] = rerouted_sol

        for u, v in zip(rerouted_sol[0:], rerouted_sol[1:]):
            """
            Update net.edge_to_sink_idx_set and edge_to_routing_net_set
            """
            e = min(u, v), max(u, v)
            if e not in net.edge_to_sink_idx_set:
                net.edge_to_sink_idx_set[e] = set()
                data = self.weighted_routing_resource_net[u][v]
                if data['type'] == 1:
                    data['weight'] += 1.0 / data['resource_cnt']
                    data['pre_adjusted_weight'] += 1.0 / data['resource_cnt']
                data['usage'] += 1

            net.edge_to_sink_idx_set[e].add(sink_idx)
            self.edge_to_routing_net_set[e].add(net)



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
                sink_idx_criticality_pair = [(sink_idx, net.sink_criticality[sink_idx])
                                             for sink_idx in range(len(net.sinks))]

                sink_idx_criticality_pair.sort(key=lambda pair: pair[1], reverse=True)

                for sink_idx, _ in sink_idx_criticality_pair:
                     ripped_up_sol = self.rip_up_sink(net, sink_idx)
                     self.reroute_sink(net, sink_idx, ripped_up_sol)

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
    # hybrid_init_routing_res_path = "/home/huqf/FPGADieRouting/Res/Baseline/testcase9/run_20241113112804/hyb_init_routing_res.pkl"
    hybrid_init_routing_res_path = "/home/huqf/FPGADieRouting/Res/Baseline/testcase10/run_20241113114751/hyb_init_routing_res.pkl"

    with open(hybrid_init_routing_res_path, "rb") as f:
        nets, weighted_routing_resource_network = pickle.load(f)

    start = time.time()
    vio_rip_rerouter = VioRipReroute(nets, weighted_routing_resource_network)
    vio_rip_rerouter.rip_up_and_reroute()
    end = time.time()
    print('Violation Rip Up and Reroute took {} seconds'.format(end - start))

    start = time.time()
    perf_driven_reroute = PerfDrivenRipRerouteSink(vio_rip_rerouter.nets,
                                                   vio_rip_rerouter.weighted_routing_resource_net,
                                                   vio_rip_rerouter.edge_to_routing_net_set)
    perf_driven_reroute.perf_driven_rip_reroute()
    end = time.time()
    print('Performance-Driven Rip Up and Reroute took {} seconds'.format(end - start))


