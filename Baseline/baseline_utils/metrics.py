from Baseline.Net.net import Net

def get_net_criticality(weighted_routing_resource_net, net: Net):
    """
    The criticality of a net is defined as the maximum criticality among all its sinks.
    :param weighted_routing_resource_net: networkx
    :param net:
    :return:
    """

    """
    Recalculate criticality.
    """
    if net.criticality is None:
        net.sink_criticality = [None] * len(net.sinks)
        for idx, sol in enumerate(net.routing_solutions):
            delay = 0
            for u, v in zip(sol[0:], sol[1:]):
                delay += weighted_routing_resource_net[u][v]['weight']
            net.sink_criticality[idx] = delay

        net.criticality = max(net.sink_criticality)

    return net.criticality