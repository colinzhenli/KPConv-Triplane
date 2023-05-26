import torch
import torch.nn as nn
import torch.nn.functional as F

def pc_normalize(pc):
    # centroid = torch.mean(pc, axis=0)
    # pc = pc - centroid
    # m = torch.max(torch.sqrt(torch.sum(pc ** 2, axis=1)))
    pc = pc / 0.4
    return pc

def knn(x, k):
    B, _, N = x.size()
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    _, idx = pairwise_distance.topk(k=k, dim=-1)  # (batch_size, num_points, k)

    return idx, pairwise_distance


def get_scorenet_input(x, idx, k):
    """(neighbor, neighbor-center)"""
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()

    neighbor = x.view(batch_size * num_points, -1)[idx, :]

    neighbor = neighbor.view(batch_size, num_points, k, num_dims)

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    xyz = torch.cat((neighbor - x, neighbor), dim=3).permute(0, 3, 1, 2)  # b,6,n,k

    return xyz


def feat_trans_dgcnn(point_input, kernel, m):
    """transforming features using weight matrices"""
    # following get_graph_feature in DGCNN: torch.cat((neighbor - center, neighbor), dim=3)
    B, _, N = point_input.size()  # b, 2cin, n
    point_output = torch.matmul(point_input.permute(0, 2, 1).repeat(1, 1, 2), kernel).view(B, N, m, -1)  # b,n,m,cout
    center_output = torch.matmul(point_input.permute(0, 2, 1), kernel[:point_input.size(1)]).view(B, N, m, -1)  # b,n,m,cout
    return point_output, center_output


def feat_trans_pointnet(point_input, kernel, m):
    """transforming features using weight matrices"""
    # no feature concat, following PointNet
    N, _ = point_input.size()  # n, cin
    point_output = torch.matmul(point_input, kernel).view(N, -1)  # n, m*cout
    return point_output


class ScoreNet(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_unit=[16], last_bn=False):
        super(ScoreNet, self).__init__()
        self.hidden_unit = hidden_unit
        self.last_bn = last_bn
        self.mlp_convs_hidden = nn.ModuleList()
        self.mlp_bns_hidden = nn.ModuleList()

        if hidden_unit is None or len(hidden_unit) == 0:
            self.mlp_convs_nohidden = nn.Conv2d(in_channel, out_channel, 1, bias=not last_bn)
            if self.last_bn:
                self.mlp_bns_nohidden = nn.BatchNorm2d(out_channel)

        else:
            self.mlp_convs_hidden.append(nn.Conv2d(in_channel, hidden_unit[0], 1, bias=False))  # from in_channel to first hidden
            self.mlp_bns_hidden.append(nn.BatchNorm2d(hidden_unit[0]))
            for i in range(1, len(hidden_unit)):  # from 2nd hidden to next hidden to last hidden
                self.mlp_convs_hidden.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1, bias=False))
                self.mlp_bns_hidden.append(nn.BatchNorm2d(hidden_unit[i]))
            self.mlp_convs_hidden.append(nn.Conv2d(hidden_unit[-1], out_channel, 1, bias=not last_bn))  # from last hidden to out_channel
            self.mlp_bns_hidden.append(nn.BatchNorm2d(out_channel))

    def forward(self, xyz, calc_scores='softmax', bias=0):
        B, _, N, K = xyz.size()
        scores = xyz

        if self.hidden_unit is None or len(self.hidden_unit) == 0:
            if self.last_bn:
                scores = self.mlp_bns_nohidden(self.mlp_convs_nohidden(scores))
            else:
                scores = self.mlp_convs_nohidden(scores)
        else:
            for i, conv in enumerate(self.mlp_convs_hidden):
                if i == len(self.mlp_convs_hidden)-1:  # if the output layer, no ReLU
                    if self.last_bn:
                        bn = self.mlp_bns_hidden[i]
                        scores = bn(conv(scores))
                    else:
                        scores = conv(scores)
                else:
                    bn = self.mlp_bns_hidden[i]
                    scores = F.relu(bn(conv(scores)))

        if calc_scores == 'softmax':
            scores = F.softmax(scores, dim=1)+bias  # B*m*N*K, where bias may bring larger gradient
        elif calc_scores == 'sigmoid':
            scores = torch.sigmoid(scores)+bias  # B*m*N*K
        else:
            raise ValueError('Not Implemented!')

        scores = scores.permute(0, 2, 3, 1)  # B*N*K*m

        return scores
    
class Mlps(nn.Module):
    """Mlps implemented as (1x1) convolution."""

    def __init__(self, inc, outc_list=[128], last_bn_norm=True):
        """Initialize network with hyperparameters.

        Args:
            inc (int): number of channels in the input.
            outc_list (List[]): list of dimensions of hidden layers.
            last_bn_norm (boolean): determine if bn and norm layer is added into the output layer.
        """
        assert len(outc_list) > 0
        super(Mlps, self).__init__()

        self.layers = nn.Sequential()

        # We compose MLPs according to the list of out_channel (`outc_list`).
        # Additionally, we use the flag `last_bn_norm` to
        # determine if we want to add norm and activation layers
        # at last layer.
        for i, outc in enumerate(outc_list):
            self.layers.add_module(f"Linear-{i}", nn.Conv2d(inc, outc, 1))
            if i + 1 < len(outc_list) or last_bn_norm:
                self.layers.add_module(f"BN-{i}", nn.BatchNorm2d(outc))
                self.layers.add_module(f"ReLU-{i}", nn.ReLU(inplace=True))
            inc = outc

    def forward(self, x, format="BCNM"):
        """Forward pass.

        Args:
            x (torch.tensor): input tensor.
            format (str): format of point tensor.
                Options include 'BCNM', 'BNC', 'BCN'
        """
        assert format in ["BNC", "BCNM", "BCN"]

        # Re-formate tensor into "BCNM".
        if format == "BNC":
            x = x.transpose(2, 1).unsqueeze(-1)
        elif format == "BCN":
            x = x.unsqueeze(-1)

        # We use the tensor of the "BCNM" format.
        x = self.layers(x)

        # Re-formate tensor back input format.
        if format == "BNC":
            x = x.squeeze(-1).transpose(2, 1)
        elif format == "BCN":
            x = x.squeeze(-1)

        return x