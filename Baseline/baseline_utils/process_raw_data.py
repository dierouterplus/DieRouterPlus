import os
import networkx as nx
from collections import OrderedDict
from utils.process_raw_data import build_die2fpga, build_edge_list, build_pin2die
from Baseline.Net.net import Net


def build_weighted_routing_resource_network(fpga_die_relation_path, die_network_path):
    """
    Build a routing resource network, wherein each node represents an individual FPGA,
    and an edge signifies the interconnection of wires.
    The resources_cnt attribute of an edge denotes the number of wires that are available for routing.
    Additionally, the type attribute of an edge indicates whether the connection spans across two dies
    residing in separate FPGAs or within the same FPGA.

    :param fpga_die_relation_path:
    :param die_network_path:
    :return:
    """
    # {die_id: fpga_id}
    die2fpga = build_die2fpga(fpga_die_relation_path)
    # [ (u, v, {'resource_cnt': resource_cnt}) ]
    edge_list = build_edge_list(die_network_path)

    """
    append attributes to the edge
    0: ends are in the same fpga
    1: ends are in different fpgas
    """
    for u, v, attr in edge_list:
        if die2fpga[u] == die2fpga[v]:
            attr['type'] = 0
            attr['weight'] = 1
        else:
            attr['type'] = 1
            attr['weight'] = 0.5
        attr['usage'] = 0

    routing_resource_network = nx.Graph()
    routing_resource_network.add_edges_from(edge_list)

    return routing_resource_network


def create_nets(netlist_file_path, pin_location_path):
    """
    Create routing tasks. Each routing task is represented by a list.
    [source_die_id, sink_die_id_1, ..., sink_die_id_k]
    :param netlist_file_path:
    :param pin_location_path:
    :return:
    """
    # {pin: die_id}
    pin2die = build_pin2die(pin_location_path)

    driver_label = 's'
    nets = []
    """
    The number of pins would be used to determine the criticality of the net.
    """
    n_pins = []
    with open(netlist_file_path, 'r') as in_file:
        net = []
        for line in in_file:
            if driver_label in line and len(net) != 0:  # Encounter a new net.
                nets.append(net)
                n_pins.append(len(net))
                net = []
            net.append(pin2die[line.strip().split()[0]])

        nets.append(net)
        n_pins.append(len(net))

    # Remove duplicated sinks (multiple pins on the same die) from each net.
    nets = [list(OrderedDict.fromkeys(_)) for _ in nets]

    """
    Instantiate the net objects.
    """
    net_objs = []
    assert len(nets) == len(n_pins)
    for i, net in enumerate(nets):
        # Remove the net that source and all sinks are on the same die.
        if len(net) > 1:
           net_objs.append(Net(net, n_pins[i], len(net_objs)))

    return net_objs


def build_weighted_routing_resource_network_and_nets(testcase_dir):
    """
    Build routing resource network and tasks from text files in testcase_dir.
    :param testcase_dir:
    :return:
    """
    fpga_die_relation_path = os.path.join(testcase_dir, 'design.fpga.die')
    die_network_path = os.path.join(testcase_dir, 'design.die.network')
    netlist_file_path = os.path.join(testcase_dir, 'design.net')
    pin_location_path = os.path.join(testcase_dir, 'design.die.position')

    weighted_routing_resource_network = build_weighted_routing_resource_network(fpga_die_relation_path,
                                                                                die_network_path)
    nets = create_nets(netlist_file_path, pin_location_path)

    return weighted_routing_resource_network, nets


if __name__ == '__main__':
    testcase_dir = '../../Data/testcase6'
    weighted_routing_resource_network, nets = build_weighted_routing_resource_network_and_nets(testcase_dir)
    print(weighted_routing_resource_network.edges(data=True))
    print(nets)