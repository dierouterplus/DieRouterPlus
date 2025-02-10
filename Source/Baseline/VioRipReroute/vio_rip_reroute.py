import pickle
import heapq
import networkx as nx
from Source.Baseline.Net.net import Net
from Source.Baseline.baseline_utils.metrics import get_net_criticality
class VioRipReroute:
    def __init__(self, nets, weighted_routing_resource_network):
        """
        :param nets: a list of Net objects.
        :param weighted_routing_resource_network:
        Attributes of an edge:
          [1] resource_cnt : number of physical wires on the edge
          [2] type: 0/1 represents internal/external edge
          [3] weight: delay of the edge
          [4] pre_adjusted_weight:
              for non-tdm edge, weight = pre_adjusted_weight = 1.0
              for tdm edge, weight + 1/usage = pre_adjusted_weight
          [5] usage: the number of nets using the edge
        """

        self.nets = nets
        self.weighted_routing_resource_net = weighted_routing_resource_network

        """
        Mapping an edge to the set of nets using that edge for routing.
        We store net in the set. A net is hashed by net_id.
        dict(e -> set(net))
        e = (u, v), u < v
        """
        self.edge_to_routing_net_set = self.init_edge_to_routing_net_set()

        """
        heap: store overflowing internal edges. list(... (-1 * overflow, e) ...)
        heap_set: elements in heap. 
        to_update: store edges in the heap that need their overflow information updated.
        """
        self.heap = []
        self.heap_set = set()
        self.to_update = set()


    def init_edge_to_routing_net_set(self):
        """
        :return: dict((u, v)->set(net_id))
        """
        edge_to_routing_net_set = {}
        for net in self.nets:
            for sol in net.routing_solutions:
                for u, v in zip(sol[0:], sol[1:]):
                    u, v = min(u, v), max(u, v)
                    e = (u, v)
                    """
                    The same edge may appear in paths leading to different pins.
                    Here, we use set. Thus, duplication is removed automatically.
                    """
                    if e not in edge_to_routing_net_set:
                        edge_to_routing_net_set[e] = set()
                    edge_to_routing_net_set[e].add(net)
        return edge_to_routing_net_set



    def init_heap(self):
        for u, v, data in self.weighted_routing_resource_net.edges(data=True):
            """
            We only put internal edges into the heap.
            """
            if data['type'] == 0:
                overflow = data['usage'] - data['resource_cnt']
                if overflow > 0:
                    e = min(u, v), max(u, v)
                    self.heap_set.add(e)
                    heapq.heappush(self.heap, (-1 * overflow, e))

    def get_heap_top(self):
        while len(self.heap) > 0:
            _, e = heapq.heappop(self.heap)
            self.heap_set.remove(e)

            """
            The overflow information on e is outdated.
            """
            if e in self.to_update:
                u, v = e
                data = self.weighted_routing_resource_net[u][v]
                overflow = data['usage'] - data['resource_cnt']
                if overflow > 0:
                    heapq.heappush(self.heap, (-1 * overflow, e))
                    self.heap_set.add(e)
                self.to_update.remove(e)
            else:
                return e

        return None




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
                self.weighted_routing_resource_net[u][v]['weight'] -= 1/self.weighted_routing_resource_net[u][v]['resource_cnt']
                self.weighted_routing_resource_net[u][v]['pre_adjusted_weight'] -= 1 / self.weighted_routing_resource_net[u][v][
                    'resource_cnt']


            """
            Step 2. Del net from edge_to_routing_net_set[edge].
            """
            e = min(u, v), max(u, v)
            self.edge_to_routing_net_set[e].remove(net)

            """
            Step 3. Update heap.
            e not in self.heap_set 
            (Case 1) e is an internal edge & no congestion on e => no congestion after ripping a net using e
            (Case 2) e is an external edge => do not need to add an external edge to the heap
            """
            if e in self.heap_set:
                self.to_update.add(e)

        """
        Step 4. Clear up related net info.
        """
        net.criticality = None
        net.sink_criticality = None
        net.routing_solutions = None

    @staticmethod
    def compute_reroute_weight(u, v, data):
        """
        :param u:
        :param v:
        :param data: attribute dict of edge (u, v)
        :return:
        """
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
                                                 weight=VioRipReroute.compute_reroute_weight)

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
                self.weighted_routing_resource_net[u][v]['weight'] += 1/self.weighted_routing_resource_net[u][v]['resource_cnt']
                self.weighted_routing_resource_net[u][v]['pre_adjusted_weight'] += 1/self.weighted_routing_resource_net[u][v]['resource_cnt']



            """
            Step 4. Update heap.
            """
            if e in self.heap_set:
                self.to_update.add(e)
            else:
                if self.weighted_routing_resource_net[u][v]['type'] == 0:
                    overflow = self.weighted_routing_resource_net[u][v]['usage'] - self.weighted_routing_resource_net[u][v]['resource_cnt']
                    if overflow > 0:
                        heapq.heappush(self.heap, (-1 * overflow, e))
                        self.heap_set.add(e)


    def rip_up_and_reroute(self):
        self.init_heap()

        while len(self.heap) > 0:
            # Extract the edge with the Maximal Overflow
            e = self.get_heap_top()

            if e is None:
                continue

            # Extract nets using e
            nets_using_e = list(self.edge_to_routing_net_set[e])


            """
            Sort nets by criticality.
            Bug Fix: Prioritize removing nets with lower criticality.
            """
            nets_using_e.sort(key=lambda net: get_net_criticality(self.weighted_routing_resource_net, net))

            for net in nets_using_e:
                self.rip_up(net)
                self.reroute(net)




if __name__ == "__main__":
    hybrid_init_routing_res_path = "/home/huqf/FPGADieRouting/Res/Baseline/testcase6/run_20241111193533/hyb_init_routing_res.pkl"

    with open(hybrid_init_routing_res_path, "rb") as f:
        nets, weighted_routing_resource_network = pickle.load(f)

    vio_rip_rerouter = VioRipReroute(nets, weighted_routing_resource_network)

    vio_rip_rerouter.rip_up_and_reroute()