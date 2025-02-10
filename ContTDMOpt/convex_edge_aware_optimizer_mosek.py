import sys

import math
import numpy as np

sys.path.append('../')

import pickle
from utils.timing import timeit
from mosek.fusion import *

class ConvexEdgeAwareOptimizer:
    def __init__(self, nets, weighted_routing_resource_net):
        """
        :param nets:
        :param weighted_routing_resource_net:
        Attributes of an edge:
          [1] resource_cnt : number of physical wires on the edge
          [2] type: 0/1 represents internal/external edge
          [3] weight: delay of the edge
          [4] pre_adjusted_weight: weight + 1/resource_cnt for TDM edge; weight for non-TDM edge.
          [5] usage: the number of nets using the edge
        """
        self.nets = nets
        self.weighted_routing_resource_net = weighted_routing_resource_net

        """
        All variables form a matrix, Var, where Var[i][j] represents the TDM ratio of the j-th net on the i-th tdm edge.
        This function maps tdm edges to row indices in the Var matrix.
        """
        self.tdm_edge_row_idx_map = self.construct_tdm_edge_row_idx_map()
        self.n_tdm_edge = len(self.tdm_edge_row_idx_map)
        self.n_nets = len(self.nets)

        """
        np.array: (n_tdm_edge, )
        rsc_constraints[i] is the resource constraint on the i-th tdm edge.
        """
        self.rsc_constraints = self.construct_rsc_constraints()



    def construct_tdm_edge_row_idx_map(self):
        tdm_edge_row_idx_map = {}
        row_idx = 0
        for u, v, data in self.weighted_routing_resource_net.edges(data=True):
            if data['type'] == 1:
                e = min(u, v), max(u, v)
                tdm_edge_row_idx_map[e] = row_idx
                row_idx += 1
        return tdm_edge_row_idx_map


    def construct_rsc_constraints(self):

        rsc_constraints = np.zeros(self.n_tdm_edge)
        for e, row_idx in self.tdm_edge_row_idx_map.items():
            rsc_constraints[row_idx] = self.weighted_routing_resource_net[e[0]][e[1]]['resource_cnt']

        return rsc_constraints


    def generate_mask(self):
        """
        return:
        tdm_edge_net_usage_matrix: (n_tdm_edge, n_nets)
        tdm_edge_net_usage_matrix[i, j] = 1: nets[j] traverses i-th tdm edge

        tdm_edge_path_usage_matrix: (n_tdm_edge, n_paths)
        tdm_edge_path_usage_matrix[i, j] = 1: j-th path traverses i-th tdm edge

        path_bias: (n_paths, )
        """
        tdm_edge_net_usages = []
        tdm_edge_path_usages = []
        path_bias = []

        for net in self.nets:
            tmd_edge_traversed_by_net = np.zeros(self.n_tdm_edge)
            for sol in net.routing_solutions:
                """
                tdm edges traversed by sol
                """
                tdm_edge_traversed_by_path = np.zeros(self.n_tdm_edge)
                bias = 0
                for u, v in zip(sol[0:], sol[1:]):
                    if self.weighted_routing_resource_net[u][v]['type'] == 1:
                        e = min(u, v), max(u, v)
                        tdm_edge_traversed_by_path[self.tdm_edge_row_idx_map[e]] = 1
                        bias += 0.5
                    else:
                        bias += 1

                tmd_edge_traversed_by_net[tdm_edge_traversed_by_path == 1] = 1
                tdm_edge_path_usages.append(tdm_edge_traversed_by_path)
                path_bias.append(bias)

            tdm_edge_net_usages.append(tmd_edge_traversed_by_net)

        """
        np.vstack(tdm_edge_net_usages) : (n_nets, n_tdm_edge) -- transpose -> (n_tdm_edge, n_nets)
        np.vstack(tdm_edge_path_usages) : (n_paths, n_tdm_edge) -- transpose -> (n_tdm_edge, n_paths)
        (n_paths, )
        """
        return np.transpose(np.vstack(tdm_edge_net_usages)), np.transpose(np.vstack(tdm_edge_path_usages)), np.array(path_bias)

    @timeit("Solve")
    def solve(self):

        """
        tdm_edge_net_usage_matrix: (n_tdm_edge, n_nets), np.Array
        tdm_edge_net_usage_matrix[i, j] = 1: nets[j] traverses i-th tdm edge

        tdm_edge_path_usage_matrix: (n_tdm_edge, n_paths), np.Array
        tdm_edge_path_usage_matrix[i, j] = 1: j-th path traverses i-th tdm edge

        path_bias: (n_paths, ), np.Array
        """
        tdm_edge_net_usage_matrix, tdm_edge_path_usage_matrix, path_bias = self.generate_mask()

        with Model() as M:
            """
            TDM Ratio >= 1
            """
            # tdm_edge_net_var_matrix = M.variable([self.n_tdm_edge, self.n_nets], Domain.greaterThan(1.))
            tdm_edge_net_var_matrix = M.variable([self.n_tdm_edge, self.n_nets], Domain.greaterThan(4.))

            var_list = []
            for col_idx, net in enumerate(self.nets):
                """
                The sliced result is tdm_edge_net_var_matrix[0: self.n_tdm_edge, col_idx] of shape (n_tdm_edge, 1)
                """
                var_list += [tdm_edge_net_var_matrix.slice([0, col_idx], [self.n_tdm_edge, col_idx+1])] * len(net.routing_solutions)

            """
            [ ..., (n_tdm_edge, 1), ... ] ---> (n_tdm_edge, n_paths)
            """
            tdm_edge_path_var_matrix = Var.hstack(var_list)

            """
            (n_tdm_edge * n_nets, )
            """
            flat_tdm_edge_net_var_matrix = Var.flatten(tdm_edge_net_var_matrix)

            """
            Auxiliary variables: (n_tdm_edge * n_nets, )
            To describe the second order conic constraint.
            2xz >= (sqrt(2))**2
            """
            aux_var = M.variable(self.n_tdm_edge * self.n_nets, Domain.greaterThan(0.))
            aux_var_1 = M.variable(self.n_tdm_edge * self.n_nets, Domain.equalsTo(math.sqrt(2)))

            """
            Calculate delay of all paths.
            (n_paths, )
            """
            delays = Expr.add(Expr.sum(Expr.mulElm(tdm_edge_path_var_matrix, tdm_edge_path_usage_matrix),
                          0), path_bias)

            """
            maximal_delay >= all path delays
            """
            maximal_delay = M.variable()
            M.constraint(Expr.sub(Var.vrepeat(maximal_delay, delays.getSize()), delays), Domain.greaterThan(0.))

            """
            second order conic constraint
            2xz >= (sqrt(2)) ** 2
            """
            conv_matrix = Expr.hstack(flat_tdm_edge_net_var_matrix, aux_var, aux_var_1)
            M.constraint(conv_matrix, Domain.inRotatedQCone())

            """
            z_1 + z_2 + ... z_n == k
            """
            M.constraint(
                Expr.sub(
                    self.rsc_constraints,
                    Expr.sum(
                        Expr.mulElm(Var.reshape(aux_var, self.n_tdm_edge, self.n_nets),
                        tdm_edge_net_usage_matrix), 1
                    )
                ),
                Domain.equalsTo(0.)
            )

            M.objective(ObjectiveSense.Minimize, maximal_delay)
            M.solve()

            """
            Numpy Array : (n_tdm_edge, n_nets)
            """
            solved_var_matrix = np.reshape(tdm_edge_net_var_matrix.level(), (self.n_tdm_edge, self.n_nets))

            # repeated_times = [len(net.routing_solutions) for net in self.nets]
            # np_obj_val = np.max(np.sum(np.repeat(solved_var_matrix, repeated_times, axis=1) * tdm_edge_path_usage_matrix,
            #                            axis=0) + path_bias)
            #
            # assert np_obj_val - M.primalObjValue() < 1e-5

            return solved_var_matrix, M.primalObjValue(), M.getPrimalSolutionStatus(), path_bias









if __name__ == '__main__':
    # HP Server
    # two_stage_reroute_res = "../Res/Baseline/testcase6/run_20241117152129/run_20241117160605/two_stage_reroute_res.pkl"
    # Lenovo Server
    # two_stage_reroute_res = "../Res/Baseline/testcase4/run_20241113111933/run_20241120170743/two_stage_reroute_res.pkl"
    # two_stage_reroute_res = "../Res/Baseline/testcase5/run_20241113111752/run_20241120170927/two_stage_reroute_res.pkl"
    # two_stage_reroute_res = "../Res/Baseline/testcase6/run_20241111193533/run_20241119093605/two_stage_reroute_res.pkl"
    two_stage_reroute_res = "../Res/Baseline/testcase7/run_20241112201514/run_20241119141119/two_stage_reroute_res.pkl"
    # two_stage_reroute_res = "../Res/Baseline/testcase8/run_20241113093433/run_20241119165208/two_stage_reroute_res.pkl"
    # two_stage_reroute_res = "../Res/Baseline/testcase9/run_20241113112804/run_20241119170429/two_stage_reroute_res.pkl"
    # two_stage_reroute_res = "../Res/Baseline/testcase10/run_20241113114751/run_20241120144955/two_stage_reroute_res.pkl"

    with open(two_stage_reroute_res, "rb") as f:
        nets, weighted_routing_resource_net, _ = pickle.load(f)

    convex_edge_aware_optimizer = ConvexEdgeAwareOptimizer(nets, weighted_routing_resource_net)
    solved_var_matrix, primal_obj_val, sol_status = convex_edge_aware_optimizer.solve()
    print(primal_obj_val, sol_status)
