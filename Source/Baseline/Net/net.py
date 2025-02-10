class Net:
    def __init__(self, net, n_pins, net_id=None):
        self.net_id = net_id
        self.n_pins = n_pins
        """
        src and sinks are die nodes rather than pins.
        """
        self.src = net[0]
        self.sinks = net[1:]
        self.criticality = None
        self.sink_criticality = None
        self.routing_solutions = None

        """
        dict((u, v) -> set(idx))
        u < v , (u, v) or (v, u) on the path from src to sinks[idx]
        Used in PerfDrivenRipReroute.perf_driven_rip_reroute_sink.py
        """
        self.edge_to_sink_idx_set = None

        """
        dict((u, v) -> list(idx))
        (u, v) is an directed tdm edge
        Used in Conti TDM Solver
        """
        self.directed_tdm_edge_to_sink_idx_list = None





    """
    Rewrite __eq__ and __hash__ methods to guarantee the consistent behavior in hash-based collections.
    """
    def __eq__(self, other):
        return self.net_id == other.net_id

    def __hash__(self):
        return hash(self.net_id)