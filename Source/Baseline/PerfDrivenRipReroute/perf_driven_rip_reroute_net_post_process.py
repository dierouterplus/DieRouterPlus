import sys
sys.path.append("../../../")
import pickle
import time
from copy import deepcopy

import math

from Source.Baseline.VioRipReroute.vio_rip_reroute import VioRipReroute
from Source.Baseline.baseline_utils.metrics import get_net_criticality
from Source.Baseline.Net.net import Net
import networkx as nx
from tqdm import tqdm

class PerfDrivenRipRerouteNet:
    def __init__(self, nets, weighted_routing_resource_net, edge_to_routing_net_set, patience=10):
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
        :param patience: If the result remains the same for patience consecutive iterations, terminate the process.
        """
        self.nets = nets
        self.weighted_routing_resource_net = weighted_routing_resource_net

        """
        dict(net_id -> net idx in self.nets)
        """
        self.net_id_to_net_idx = {}
        for net_idx, net in enumerate(self.nets):
            self.net_id_to_net_idx[net.net_id] = net_idx


        """
        dict(edge -> set(net idx))
        """
        self.edge_to_routing_net_id_set = {}
        for edge, routing_net_set in edge_to_routing_net_set.items():
            net_id_set = set()
            for net in routing_net_set:
                net_id_set.add(net.net_id)
            self.edge_to_routing_net_id_set[edge] = net_id_set



        self.system_delay = max([get_net_criticality(self.weighted_routing_resource_net, net) for net in self.nets])
        print('System delay is {}'.format(self.system_delay))

        """
        Store the optimal results obtained during the iteration process.
        """
        self.opt_system_delay = math.inf
        self.opt_nets = None
        self.opt_weighted_routing_resource_net = None
        self.opt_edge_to_routing_net_id_set = None

        self.patience = patience



    def get_edge_criticality(self, u, v):
        """
        edge_criticality is the sum of the criticality of all nets passing through it.
        :param u:
        :param v:
        :return:
        """
        e = min(u, v), max(u, v)

        if len(self.edge_to_routing_net_id_set[e]) == 0:
            return 0.0

        net_criticality = [get_net_criticality(self.weighted_routing_resource_net,
                                               self.nets[self.net_id_to_net_idx[net_id]])
                           for net_id in self.edge_to_routing_net_id_set[e]]

        edge_criticality = max(net_criticality)

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


    def extract_critical_edges(self):

        critical_edges = []
        for u, v, data in self.weighted_routing_resource_net.edges(data=True):
            if data['type'] == 1:
                criticality = self.get_edge_criticality(u, v)
                e = min(u, v), max(u, v)
                critical_edges.append((e, criticality))

        critical_edges.sort(key=lambda x: x[1], reverse=True)
        return critical_edges

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
            Step 2. Del net id from edge_to_routing_net_id_set[edge].
            """
            e = min(u, v), max(u, v)
            self.edge_to_routing_net_id_set[e].remove(net.net_id)

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
            self.edge_to_routing_net_id_set[e].add(net.net_id)

            """
            Step 3. Update weighted_routing_resource_network
            """
            self.weighted_routing_resource_net[u][v]['usage'] += 1
            if self.weighted_routing_resource_net[u][v]['type'] == 1:
                self.weighted_routing_resource_net[u][v]['weight'] += 1 / self.weighted_routing_resource_net[u][v]['resource_cnt']
                self.weighted_routing_resource_net[u][v]['pre_adjusted_weight'] += 1 / self.weighted_routing_resource_net[u][v]['resource_cnt']

    def perf_driven_rip_reroute(self):

        prev_system_delay = 0.0
        repeated_system_delay_cnt = 0

        while True:
            e = self.extract_most_critical_edge()

            """
            Calculate the nets that need to be removed, and sort them in descending order based on criticality.
            """
            # nets_to_rip_up = list(self.edge_to_routing_net_set[e])
            nets_to_rip_up = [self.nets[self.net_id_to_net_idx[net_id]]
                              for net_id in self.edge_to_routing_net_id_set[e]]

            nets_to_rip_up.sort(key=lambda net: get_net_criticality(self.weighted_routing_resource_net, net),
                                reverse=True)

            for net in tqdm(nets_to_rip_up):
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
                self.opt_edge_to_routing_net_id_set = deepcopy(self.edge_to_routing_net_id_set)



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

        return (self.opt_nets, self.opt_weighted_routing_resource_net, self.opt_edge_to_routing_net_id_set,
                self.opt_system_delay)


    def perf_post_process(self, nets, weighted_routing_resource_net, edge_to_routing_net_id_set, system_delay):
        """
        :param nets:
        :param weighted_routing_resource_net:
        :param edge_to_routing_net_id_set:
        :param system_delay:
        :return:
        """

        self.nets = nets
        self.weighted_routing_resource_net = weighted_routing_resource_net
        self.edge_to_routing_net_id_set = edge_to_routing_net_id_set
        self.system_delay = system_delay

        """
        Processing Critical Edge e
        - Iteratively reroute subsets of nets passing through e to avoid the increase in system delay 
        caused by rerouting all nets at once. 
        - The processing of critical edge e ends when either: (a) the system delay decreases, or 
        (b) all nets on e have been rerouted. 

        Processing the List of Critical Edges
        - After completing the processing of edge e:
        (a) If there is no decrease in system delay, proceed to process the next edge in the list of critical edges; 
        (b) If there is a decrease in system latency, regenerate a new list of critical edges.

        Termination Condition: 
        - The program terminates after traversing the entire list of critical edges if 
        no further operations can reduce the system delay.
        """
        is_updated = True # Indicate the system delay or not.
        critical_edges = None
        n_critical_edge = 0 # The processed critical edges.
        while True:
            if is_updated:
                """
                Regenerate critical edges.
                """
                critical_edges = self.extract_critical_edges()
                e = critical_edges[0][0]
                n_critical_edge = 0
            else:
                """
                Process next critical edge.
                """
                n_critical_edge += 1
                if n_critical_edge >= len(critical_edges):
                    break
                e = critical_edges[n_critical_edge][0]

            print(f"Process edge {e}")

            """
            Calculate the nets that need to be removed, and sort them in ascending order based on criticality.
            
            e -> net_id_set -> index -> net
            """
            net_id_to_rip_up = list(self.edge_to_routing_net_id_set[e])
            net_id_to_rip_up.sort(key=lambda net_id: get_net_criticality(self.weighted_routing_resource_net,
                                                                         self.nets[self.net_id_to_net_idx[net_id]])
                                  )

            """
            If the rerouting results in a worse outcome, revert to the previous routing.
            The reason for using edge_to_routing_net_id_set instead of edge_to_routing_net_set is that after performing 
            a deep copy of nets, the net objects in copied_nets are different from those in nets. 
            However, edge_to_routing_net_set records the addresses of objects in nets, not in copied_nets.
            """
            copied_nets = deepcopy(self.nets)
            copied_weighted_routing_resource_net = deepcopy(self.weighted_routing_resource_net)
            copied_edge_to_routing_net_id_set = deepcopy(self.edge_to_routing_net_id_set)
            copied_net_id_to_rip_up = deepcopy(net_id_to_rip_up)

            """
            In the rerouting process, if the routing of threshold nets changes, 
            the delays of all nets are recalculated.
            """
            counter = 0
            threshold = 100
            is_updated = False
            """
            After recalculating the delays of all nets, if the system delay does not decrease, 
            continue processing the nets on critical edge e starting from net_id_idx.
            """
            net_id_idx = 0
            n_rerouted_net = 0
            while net_id_idx < len(net_id_to_rip_up):
                for net_id in net_id_to_rip_up[net_id_idx:]:
                    """
                    sink_idx is an index in net.sinks. 
                    Here, the indices are sorted based on the sinks' criticality.
                    """

                    net = self.nets[self.net_id_to_net_idx[net_id]]

                    criticality = net.criticality

                    prev_sols = net.routing_solutions
                    self.rip_up(net)

                    """
                    Increase the cost of edge e to encourage the net to use other edges for routing.
                    """
                    slack = self.system_delay - criticality
                    self.weighted_routing_resource_net[e[0]][e[1]]['pre_adjusted_weight'] += slack
                    self.reroute(net)
                    self.weighted_routing_resource_net[e[0]][e[1]]['pre_adjusted_weight'] -= slack

                    n_rerouted_net += 1

                    """
                    If the routing of the net has not changed, proceed to process the next net.
                    counter represents the number of rerouted nets, i.e., the net whose route has been changed.
                    """
                    if prev_sols == net.routing_solutions:
                        continue
                    else:
                        counter += 1

                    """
                    If the total number of rerouted net reach the threshold, to evaluate the system delay.
                    """
                    if counter >= threshold:
                        break

                """
                Recalculate criticality.
                """
                for net_ in self.nets:
                    net_.criticality = None
                system_delay = max(
                    [get_net_criticality(self.weighted_routing_resource_net, net_) for net_ in self.nets])

                if system_delay >= self.system_delay:
                    """
                    Restore previous path
                    Bug Fix : Use deepcopy to avoid the copied being modified~
                    """
                    print("Restore previous path")
                    self.weighted_routing_resource_net = deepcopy(copied_weighted_routing_resource_net)
                    self.nets = deepcopy(copied_nets)
                    """
                    Bug Fix: Do not forget to recover~
                    """
                    self.edge_to_routing_net_id_set = deepcopy(copied_edge_to_routing_net_id_set)
                    net_id_to_rip_up = deepcopy(copied_net_id_to_rip_up)
                    counter = 0
                    net_id_idx = n_rerouted_net
                else:
                    print("Accept new routing path")
                    self.system_delay = system_delay
                    is_updated = True

                print('System delay is {}'.format(self.system_delay))

                if is_updated is True:
                    break

        return self.nets, self.weighted_routing_resource_net, self.system_delay







if __name__ == '__main__':
    # hybrid_init_routing_res_path = "/home/huqf/FPGADieRouting/Res/Baseline/testcase4/run_20241113111933/hyb_init_routing_res.pkl"
    # hybrid_init_routing_res_path = "/home/huqf/FPGADieRouting/Res/Baseline/testcase5/run_20241113111752/hyb_init_routing_res.pkl"
    # hybrid_init_routing_res_path = "/home/huqf/FPGADieRouting/Res/Baseline/testcase6/run_20241111193533/hyb_init_routing_res.pkl"
    # hybrid_init_routing_res_path = "/home/huqf/FPGADieRouting/Res/Baseline/testcase6/run_20241227113022/hyb_init_routing_res.pkl"
    hybrid_init_routing_res_path = "/home/huqf/FPGADieRouting/Res/Baseline/testcase7/run_20241112201514/hyb_init_routing_res.pkl"
    # hybrid_init_routing_res_path = "/home/huqf/FPGADieRouting/Res/Baseline/testcase8/run_20241113093433/hyb_init_routing_res.pkl"
    # hybrid_init_routing_res_path = "/home/huqf/FPGADieRouting/Res/Baseline/testcase9/run_20241113112804/hyb_init_routing_res.pkl"
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

    opt_nets, opt_weighted_routing_resource_net, opt_edge_to_routing_net_id_set, opt_system_delay = perf_driven_reroute.perf_driven_rip_reroute()
    opt_nets, opt_weighted_routing_resource_net, opt_system_delay = perf_driven_reroute.perf_post_process(opt_nets, opt_weighted_routing_resource_net,
                                          opt_edge_to_routing_net_id_set, opt_system_delay)
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







