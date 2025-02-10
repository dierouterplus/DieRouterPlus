import os

import networkx as nx
import re
from collections import OrderedDict


def build_die2fpga(fpga_die_relation_path):
    """
    Build a map {die_id: fpga_id}.
    :param fpga_die_relation_path:
    :return:
    """
    seperators = r"[: ]"

    die2fpga = {}
    with open(fpga_die_relation_path, 'r') as in_file:
        for line in in_file:
            names = re.split(seperators, line.strip())
            fpga_name = names[0]  # FPAGXXX
            fgpa_id = int(fpga_name[4:])
            for die_name in names[1:]:  # DieXXX
                die_id = int(die_name[3:])
                die2fpga[die_id] = fgpa_id
    return die2fpga


def build_edge_list(die_network_path):
    """
    Build an edge list
    [
        (u, v, {'resource_cnt': resource_cnt})
    ]
    :param die_network_path:
    :return:
    """
    edge_list = []
    with open(die_network_path, 'r') as in_file:
        for src_die_idx, line in enumerate(in_file):
            weights = line.strip().split()
            for target_die_idx in range(src_die_idx + 1, len(weights)):
                resource_cnt = int(weights[target_die_idx])
                if resource_cnt > 0:
                    edge_list.append((src_die_idx,
                                      target_die_idx,
                                      {'resource_cnt': resource_cnt}))

    return edge_list


def build_routing_resource_network(fpga_die_relation_path, die_network_path):
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

    # append an attribute to the edge
    # 0 : ends are in the same fpga
    # 1 : ends are in the different fpga
    for u, v, attr in edge_list:
        if die2fpga[u] == die2fpga[v]:
            attr['type'] = 0
        else:
            attr['type'] = 1

    routing_resource_network = nx.Graph()
    routing_resource_network.add_edges_from(edge_list)

    return routing_resource_network


def build_pin2die(pin_location_path):
    """
    Build a map {pin: die_id}.
    :param pin_location_path:
    :return:
    """
    seperator = r"[: ]"

    pin2die = {}
    with open(pin_location_path, 'r') as in_file:
        for line in in_file:
            names = re.split(seperator, line.strip())
            die_name = names[0]  # DieXXX
            die_id = int(die_name[3:])
            for pin_name in names[1:]:
                if len(pin_name) == 0:
                    continue
                pin2die[pin_name] = die_id

    return pin2die


def create_routing_tasks(netlist_file_path, pin_location_path):
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
    with open(netlist_file_path, 'r') as in_file:
        net = []
        for line in in_file:
            if driver_label in line and len(net) != 0:  # Encounter a new net.
                nets.append(net)
                net = []
            net.append(pin2die[line.strip().split()[0]])

        nets.append(net)

    # Remove duplicated sinks (multiple pins on the same die) from each net.
    nets = [list(OrderedDict.fromkeys(_)) for _ in nets]

    # Remove the net that source and all sinks are on the same die.
    return [_ for _ in nets if len(_) > 1]


def build_routing_resource(testcase_dir):
    """
    Build routing resource network and tasks from text files in testcase_dir.

    :param testcase_dir:
    :return:
    """
    fpga_die_relation_path = os.path.join(testcase_dir, 'design.fpga.die')
    die_network_path = os.path.join(testcase_dir, 'design.die.network')
    return build_routing_resource_network(fpga_die_relation_path, die_network_path)


def build_routing_resource_and_tasks(testcase_dir):
    """
    Build routing resource network and tasks from text files in testcase_dir.

    :param testcase_dir:
    :return:
    """
    fpga_die_relation_path = os.path.join(testcase_dir, 'design.fpga.die')
    die_network_path = os.path.join(testcase_dir, 'design.die.network')
    netlist_file_path = os.path.join(testcase_dir, 'design.net')
    pin_location_path = os.path.join(testcase_dir, 'design.die.position')

    routing_resource_network = build_routing_resource_network(fpga_die_relation_path, die_network_path)
    routing_tasks = create_routing_tasks(netlist_file_path, pin_location_path)

    return routing_resource_network, routing_tasks


if __name__ == '__main__':
    testcase_dir = '../Data/testcase1'
    routing_resource_network, routing_tasks = build_routing_resource_and_tasks(testcase_dir)
    print(routing_resource_network.edges(data=True))
    print(routing_tasks)