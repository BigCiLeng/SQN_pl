import time

import torch
import torch.nn as nn

# try:
#     from torch_points import knn
# except (ModuleNotFoundError, ImportError):
from torch_points_kernels import knn, three_interpolate, three_nn
from utils.tools import DataProcessing
class SharedMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        transpose=False,
        padding_mode='zeros',
        bn=False,
        activation_fn=None
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding_mode=padding_mode
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        r"""
            Forward pass of the network

            Parameters
            ----------
            input: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, K)
        """
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Module):
    def __init__(self, d, num_neighbors):
        super(LocalSpatialEncoding, self).__init__()

        self.num_neighbors = num_neighbors
        self.mlp = SharedMLP(10, d, bn=True, activation_fn=nn.ReLU())

    def forward(self, coords, features, knn_output):
        r"""
            Forward pass

            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: torch.Tensor, shape (B, d, N, 1)
                features of the point cloud
            neighbors: tuple

            Returns
            -------
            torch.Tensor, shape (B, 2*d, N, K)
        """
        # finding neighboring points
        idx, dist = knn_output
        B, N, K = idx.size()
        # idx(B, N, K), coords(B, N, 3)
        # neighbors[b, i, n, k] = coords[b, idx[b, n, k], i] = extended_coords[b, i, extended_idx[b, i, n, k], k]
        extended_idx = idx.unsqueeze(1).expand(B, 3, N, K)
        extended_coords = coords.transpose(-2,-1).unsqueeze(-1).expand(B, 3, N, K)
        neighbors = torch.gather(extended_coords.cpu(), 2, extended_idx) # shape (B, 3, N, K)
        # if USE_CUDA:
        #     neighbors = neighbors.cuda()

        # relative point position encoding
        neighbors = neighbors.to(extended_coords)
        dist = dist.to(extended_coords)
        concat = torch.cat((
            extended_coords,
            neighbors,
            extended_coords - neighbors,
            dist.unsqueeze(-3)
        ), dim=-3)
        return torch.cat((
            self.mlp(concat),
            features.expand(B, -1, N, K)
        ), dim=-3)



class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Softmax(dim=-2)
        )
        self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.ReLU())

    def forward(self, x):
        r"""
            Forward pass

            Parameters
            ----------
            x: torch.Tensor, shape (B, d_in, N, K)

            Returns
            -------
            torch.Tensor, shape (B, d_out, N, 1)
        """
        # computing attention scores
        scores = self.score_fn(x.permute(0,2,3,1)).permute(0,3,1,2)

        # sum over the neighbors
        features = torch.sum(scores * x, dim=-1, keepdim=True) # shape (B, d_in, N, 1)

        return self.mlp(features)



class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors):
        super(LocalFeatureAggregation, self).__init__()

        self.num_neighbors = num_neighbors

        self.mlp1 = SharedMLP(d_in, d_out//2, activation_fn=nn.LeakyReLU(0.2))
        self.mlp2 = SharedMLP(d_out, 2*d_out)
        self.shortcut = SharedMLP(d_in, 2*d_out, bn=True)

        self.lse1 = LocalSpatialEncoding(d_out//2, num_neighbors)
        self.lse2 = LocalSpatialEncoding(d_out//2, num_neighbors)

        self.pool1 = AttentivePooling(d_out, d_out//2)
        self.pool2 = AttentivePooling(d_out, d_out)

        self.lrelu = nn.LeakyReLU()

    def forward(self, coords, features):
        r"""
            Forward pass

            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: torch.Tensor, shape (B, d_in, N, 1)
                features of the point cloud

            Returns
            -------
            torch.Tensor, shape (B, 2*d_out, N, 1)
        """
        knn_output = knn(coords.cpu().contiguous(), coords.cpu().contiguous(), self.num_neighbors)

        x = self.mlp1(features)

        x = self.lse1(coords, x, knn_output)
        x = self.pool1(x)

        x = self.lse2(coords, x, knn_output)
        x = self.pool2(x)

        return self.lrelu(self.mlp2(x) + self.shortcut(features))



class SQN(nn.Module):
    def __init__(self, d_in, num_classes, is_training, num_points, num_neighbors=16, decimation=4):
        super(SQN, self).__init__()
        self.num_neighbors = num_neighbors
        self.decimation = decimation
        self.is_training = is_training
        self.num_points = num_points
        self.fc_start = nn.Linear(d_in, 8)
        self.bn_start = nn.Sequential(
            nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        )

        # encoding layers
        self.encoder = nn.ModuleList([
            LocalFeatureAggregation(8, 16, num_neighbors),
            LocalFeatureAggregation(32, 64, num_neighbors),
            LocalFeatureAggregation(128, 128, num_neighbors),
            LocalFeatureAggregation(256, 256, num_neighbors)
        ])

        self.mlp = SharedMLP(512, 512, activation_fn=nn.ReLU())

        # decoding layers
        decoder_kwargs = dict(
            transpose=True,
            bn=True,
            activation_fn=nn.ReLU()
        )
        self.decoder = nn.ModuleList([
            SharedMLP(928, 256, **decoder_kwargs),
            SharedMLP(256, 128, **decoder_kwargs),
            SharedMLP(128, 64, **decoder_kwargs)
        ])

        # final semantic prediction
        self.fc_end = nn.Sequential(
            SharedMLP(64, 64, bn=True, activation_fn=nn.ReLU()),
            SharedMLP(64, 32, bn=True, activation_fn=nn.ReLU()),
            nn.Dropout(),
            SharedMLP(32, num_classes)
        )

    def forward(self, input):
        r"""
            Forward pass

            Parameters
            ----------
            input: torch.Tensor, shape (B, N, d_in)
                input points

            Returns
            -------
            torch.Tensor, shape (B, num_classes, N)
                segmentation scores for each point
        """
        N = input['points'].size(1)
        d = self.decimation
        feature = input['points']
        batch_anno_xyz = input['xyz_with_anno'].clone()
        coords = feature[...,:3]
        if self.is_training:
            feature = torch.cat([feature, DataProcessing.data_augment(feature, self.num_points)], dim=0)
            batch_anno_xyz = torch.cat([batch_anno_xyz, batch_anno_xyz], dim=0)
            coords = torch.cat([coords, coords], dim=0)

        feature = self.fc_start(feature).transpose(-2,-1).unsqueeze(-1)
        feature = self.bn_start(feature) # shape (B, d, N, 1)

        decimation_ratio = 1

        # <<<<<<<<<< ENCODER
        # f_encoder_list = []
        f_interp = []
        permutation = torch.randperm(N)
        coords = coords[:,permutation]
        feature = feature[:,:,permutation]

        for lfa in self.encoder:
            # at iteration i, x.shape = (B, N//(d**i), d_in)
            feature = lfa(coords[:,:N//decimation_ratio], feature)
            # f_encoder_list.append(feature)
            # feature_stack.append(feature.clone())
            decimation_ratio *= d
            feature = feature[:,:,:N//decimation_ratio]

            # ###########################Semantic Query############################
            dist, idx = three_nn(coords, batch_anno_xyz)
            # neighbor_xyz = DataProcessing.gather_neighbour(coords, idx)
            # xyz_tile = torch.tile(torch.unsqueeze(batch_anno_xyz, dim=2), (1, 1, idx.shape[-1], 1))
            # relative_xyz = xyz_tile - neighbor_xyz
            # dist = torch.sum(torch.square(relative_xyz), dim=-1, keepdim=False)
            weight = torch.ones_like(dist) / 3.0
            interpolated_points = three_interpolate(torch.squeeze(feature, dim=-1).contiguous(), idx, weight)
            f_interp.append(interpolated_points)

        # # # >>>>>>>>>> ENCODER

        # x = self.mlp(x)

        # <<<<<<<<<< DECODER
        interpolated_points  = torch.cat(f_interp, dim=1).unsqueeze(-1)
        for mlp in self.decoder:
            # neighbors, _ = knn(
            #     coords[:,:N//decimation_ratio].cpu().contiguous(), # original set
            #     coords[:,:d*N//decimation_ratio].cpu().contiguous(), # upsampled set
            #     1
            # ) # shape (B, N, 1)

            # extended_neighbors = neighbors.unsqueeze(1).expand(-1, x.size(1), -1, 1)

            # x_neighbors = torch.gather(x.cpu(), -2, extended_neighbors)

            # temp = x_stack.pop()
            # x_neighbors = x_neighbors.to(temp)
            # x = torch.cat((x_neighbors, temp), dim=1)

            interpolated_points = mlp(interpolated_points)

            # decimation_ratio //= d

        # >>>>>>>>>> DECODER
        # inverse permutation
        interpolated_points = interpolated_points[:,:,torch.argsort(permutation)]

        scores = self.fc_end(interpolated_points)

        return scores.squeeze(-1)


if __name__ == '__main__':
    import time
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    d_in = 7
    cloud = 1000*torch.randn(1, 2**16, d_in).to(device)
    model = SQN(d_in, 6, 16, 4, device)
    # model.load_state_dict(torch.load('checkpoints/checkpoint_100.pth'))
    model.eval()

    t0 = time.time()
    pred = model(cloud)
    t1 = time.time()
    # print(pred)
    print(t1-t0)
