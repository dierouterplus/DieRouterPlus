import math


class DPDiscretizationMixin:
    def construct_assignment_scheme(self, n_wire=None):
        """
        :param n_wire: The number of wires to construct the assignment scheme.
        :return:
        """

        """
        n_wire must not exceed the total resources available.
        """
        if n_wire is not None:
            assert n_wire <= self.resource_count
        else:
            n_wire = self.resource_count

        """
        No allocation of resources.
        """
        if n_wire == 0:
            return None

        n_unassigned_nets = self.n_net
        tdm_ratio_lst = []
        while n_unassigned_nets > 0:
            """
            Assign the last n_allocated_nets nets to a single wire.
            """
            n_allocated_nets = self.ass[n_unassigned_nets - 1, n_wire - 1]
            if n_allocated_nets > 0:
                tdm_ratio_lst = [self.r(n_allocated_nets)] * n_allocated_nets + tdm_ratio_lst

            n_unassigned_nets -= n_allocated_nets
            n_wire -= 1

        net_discrete_tdm_ratio = {}
        for idx, tdm_ratio in enumerate(tdm_ratio_lst):
            net = self.net_conti_tdm_ratio_pairs[idx][0]
            net_discrete_tdm_ratio[net] = tdm_ratio

        return net_discrete_tdm_ratio


    def r(self, n_nets):
        """
        The TDM ratio can be 1 or an integer multiple of 4.
        :param n_nets:
        :return:
        """
        # if n_nets == 1:
        #     return 1
        # else:

        return math.ceil(n_nets / 4) * 4
