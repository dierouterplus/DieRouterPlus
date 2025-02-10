from Source.Baseline.Discretization.dp_discretization_mixin import DPDiscretizationMixin
from Source.utils.timing import timeit
import torch
class DPDiscretizationTorch(DPDiscretizationMixin):
    def __init__(self, net_to_conti_tdm_ratio, resource_count):
        """
        :param net_to_conti_tdm_ratio: dict(net -> continuous tdm ratio)
        :param resource_count: number of available tdm type wires
        """
        self.net_conti_tdm_ratio_pairs = list(net_to_conti_tdm_ratio.items())
        self.resource_count = resource_count

        """
        r_0 <= r_1 <= ... <= r_{n_net-1}
        """
        self.net_conti_tdm_ratio_pairs.sort(key=lambda x: x[1])
        self.n_net = len(self.net_conti_tdm_ratio_pairs)

        """
        (n_net-1, 1)
        [
            [r_1]
            [r_2]
            ...
            [r_{n_net-1}]
        ]
        """
        self.conti_tdm_ratio_array = torch.tensor([self.net_conti_tdm_ratio_pairs[idx][1]
                                               for idx in range(1, self.n_net)], device=device)
        self.conti_tdm_ratio_array = self.conti_tdm_ratio_array.unsqueeze(-1)

        """
        dp[i][j]: the objective value of net 0 ~ i obtained by assigning
        nets 0 to i to physical wires 0 to j.
        
        [                0                     resource_count - 1
         0          [ dp[0, 0], ...,       dp[0, resource_count - 1]       ]
         1          [ dp[1, 0], ...,       dp[1, resource_count - 1]       ]
                    ...
         n_net-1    [ dp[n_net-1, 0], ..., dp[n_net-1, resource_count - 1] ]
        ]
        """
        self.dp = torch.zeros((self.n_net, self.resource_count), device=device)
        self.dp[0] = self.r(1) - self.net_conti_tdm_ratio_pairs[0][1]

        """
        (n_net, )
        [ r(1), r(2), ..., r(n_net) ]
        """
        self.r_array = torch.tensor([self.r(k) for k in range(1, self.n_net + 1)], device=device, dtype = torch.int)
        self.dp[:, 0] = self.r_array - self.net_conti_tdm_ratio_pairs[0][1]

        """
        r_matrix: (n_net-1, n_net-1)
        [               0       1     2         n_net - 2
        0            [ r(1),  r(2), r(3), ..., r(n_net-1) ]
        1            [ inf,   r(1), r(2), ..., r(n_net-2) ]
        2            [ inf,   inf,  r(1), ..., r(n_net-3) ]
        3            [ inf,   inf,  inf,  ..., r(n_net-4) ]
        ...          ...
        n_net - 2    [ inf,   inf,  inf,  ..., r(1) ]
        ]
        
        r_array: (n_net, )
           0      1        n_net-2      n_net-1 
        [ r(1), r(2), ..., r(n_net-1), r(n_net) ]
        """
        self.r_matrix = torch.full((self.n_net - 1, self.n_net - 1), torch.inf, device=device)
        for row_idx in range(0, self.n_net - 1):
            self.r_matrix[row_idx, row_idx:] = self.r_array[0: self.n_net - 1 - row_idx]

        """
        ass[i][j]: In state (i, j), the last ass[i][j] nets (nets i - ass[i][j] + 1 to i) are assigned to the j-th physical wire.
        """
        self.ass = torch.zeros((self.n_net, self.resource_count), dtype=torch.int, device=device)
        self.ass[:, 0] = torch.tensor([k for k in range(1, self.n_net + 1)], device=device, dtype=torch.int)



    @timeit("DP")
    def perform_dp(self):
        """
        :return: cost vector according to resource count 1 ~ self.resource_count.
        """

        """
        self.r_matrix : (n_net-1, n_net-1)
        [               0       1     2         n_net - 2
        0            [ r(1),  r(2), r(3), ..., r(n_net-1) ]
        1            [ inf,   r(1), r(2), ..., r(n_net-2) ]
        2            [ inf,   inf,  r(1), ..., r(n_net-3) ]
        3            [ inf,   inf,  inf,  ..., r(n_net-4) ]
        ...          ...
        n_net - 2    [ inf,   inf,  inf,  ..., r(1) ]
        ]
        
        self.conti_tdm_ratio_array : (n_net-1, 1)
        [
            [r_1]
            [r_2]
            ...
            [r_{n_net-1}]
        ]
        
        remaining_assignment_cost : (n_net-1, n_net-1)
                        assign net    0~1            0~2         0~3              0~n_net-1
                    [   col_idx        0              1           2               n_net - 2   ]
        Net Num in 
        last wire    row_idx
        col_idx + 1     0            [ r(1) - r_1,  r(2) - r_1, r(3) - r_1, ..., r(n_net-1) - r_1  ]
        col_idx         1            [ inf,         r(1) - r_2, r(2) - r_2, ..., r(n_net-2) - r_2  ]
        col_idx - 1     2            [ inf,         inf,        r(1) - r_3, ..., r(n_net-3) - r_3  ]
        col_idx - 2     3            [ inf,         inf,        inf,        ..., r(n_net-4) - r_4  ]
        ...             ...          ...
        1               n_net - 2    [ inf,         inf,        inf,        ..., r(1) - r_{n_net-1}]
        ]
        """
        remaining_assignment_cost = self.r_matrix - self.conti_tdm_ratio_array

        """
        Cost vector when assigning nets to wire 0 ~ j 
        """
        for j in range(1, self.resource_count):
            """
            self.dp[:-1, j-1].unsqueeze(-1)
            [                                   
             0          [ dp[0, j-1] ]      --- the cost of assigning net 0 to wire 0~j-1
             1          [ dp[1, j-1] ]      --- the cost of assigning net 0~1 to wire 0~j-1
             ...
             n_net-2    [ dp[n_net-2, j-1] ] --- the cost of assigning net 0~{n_net}-2 to wire 0~j-1
            ]
        
            remaining_assignment_cost : (n_net-1, n_net-1)
                        assign net    0~j          0~j+1         0~j+2              0~n_net-1
                    [   col_idx       j-1            j            j+1                n_net-2  ]
        Net Num in 
        last wire    row_idx
        col_idx + 1     0        [ r(j) - r_1 ,   r(j+1) - r_1   ...                r(n_net-1) - r_1 ]
        col_idx         1        [ r(j-1) - r_2,  r(j) - r_2     ...                r(n_net-2) - r_2 ]
                       ...            ...
        col_idx - j    j+1       [ r(1) - r_j ,   r(2) - r_j     ...                                 ]
          ...          ...            ...
           1         n_net-2     [ inf,         inf,        inf,  ...,             r(1) - r_{n_net-1}]
        ]
            """
            assignment_cost = torch.maximum(self.dp[:-1, j-1].unsqueeze(-1),
                                         remaining_assignment_cost[:, j-1:])

            """
            opt_assignment_cost:  (n_net-j, )
            optimal assignment cost of assigning net 0 ~ j, 0 ~ j+1, 0 ~ j+2, 0 ~ n_net-1
            to wire 0 ~ j.
            opt_indices: (n_net-j, )
            net num in last wire  = col_idx + 1 - opt_index 
            """
            opt_assignment_cost, opt_indices = torch.min(assignment_cost, dim=0)

            self.dp[j:, j] = opt_assignment_cost
            self.dp[j, j+1:] = self.dp[j, j]

            self.ass[j:, j] = torch.tensor(range(j, self.n_net), device=device) - opt_indices
            self.ass[j, j+1: ] = 0


        return self.dp[self.n_net - 1]



if __name__ == '__main__':

    device = torch.device("cuda")
    n_net = 2
    resource_cnt = 2
    import random
    random.seed(0)
    tdm_ratios = [random.uniform(0, 10) for _ in range(n_net)]
    net_to_tdm_ratio = dict(zip(range(n_net), tdm_ratios))
    dp = DPDiscretizationTorch(net_to_tdm_ratio, resource_cnt)
    res = dp.perform_dp()
    print(torch.mean(res), torch.min(res), torch.max(res))
    # print(dp.construct_assignment_scheme(3))





