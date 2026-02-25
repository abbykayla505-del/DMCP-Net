import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn
import random
from typing import Optional
from torch.nn import Parameter
from torch.optim.lr_scheduler import StepLR

class GCNN_net(nn.Module):
    def __init__(self, start_channels,in_channels, hidden_channels):
        super(GCNN_net, self).__init__()

        # build a 2-layer encoder

        self.conv1 = torch_geometric.nn.TransformerConv(start_channels, in_channels // 2, heads=2, bias=True,edge_dim=1)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv2 = torch_geometric.nn.TransformerConv(in_channels, hidden_channels // 2, heads=2, bias=True,edge_dim=1)
        self.bn2 = nn.BatchNorm1d(hidden_channels)

    def forward(self,data):

        x1, edge_index1, edge_attr1= data.x, data.edge_index, data.edge_attr

        x1 = self.conv1(x1, edge_index1, edge_attr1)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)

        x1 = self.conv2(x1, edge_index1, edge_attr1)
        x1 = self.bn2(x1)
        x1 = F.relu(x1)

        return x1

class GCNNet_Autoencoder(nn.Module):
    def __init__(self, start_channels,in_channels, hidden_channels):
        super(GCNNet_Autoencoder, self).__init__()

        # build a 2-layer encoder
        self.encoder=GCNN_net(start_channels,in_channels, hidden_channels)

        # build a 2-layer projector

        self.conv3 = torch_geometric.nn.TransformerConv(hidden_channels, in_channels // 2, heads=2, bias=True,
                                                        edge_dim=1)
        self.bn3 = nn.BatchNorm1d(in_channels)
        self.conv4 = torch_geometric.nn.TransformerConv(in_channels, start_channels // 2, heads=2, bias=True,
                                                        edge_dim=1)

    def forward(self, data_batch):

        yi_list=[]
        zi_list=[]

        for i in range(len(data_batch)):
            zi=self.encoder(data_batch[i])
            edge_index1, edge_attr1 = data_batch[i].edge_index, data_batch[i].edge_attr
            yi = self.conv3(zi, edge_index1, edge_attr1)
            yi = self.bn3(yi)
            yi = F.relu(yi)
            yi = self.conv4(yi, edge_index1, edge_attr1)
            yi_list.append(yi)
            zi_list.append(zi)

        yi_tensor = torch.cat(yi_list, dim=0)
        ref_group_zi = torch.sum(torch.stack(zi_list), dim=0) / len(data_batch)

        return yi_tensor,ref_group_zi

class Weighted_layer(nn.Module):

    def __init__(self, cluster_center_dim,nhead_num,modality_num):
        super(Weighted_layer, self).__init__()

        # build a Regional Heterogeneity Attention Module
        self.multiheadattention_layer=nn.MultiheadAttention(embed_dim=cluster_center_dim, num_heads=nhead_num, batch_first=True)
        self.proj_layer=nn.Linear(cluster_center_dim, modality_num)
        self.activation_layer=nn.Sigmoid()

    def forward(self, query,key,value):

        weight, _=self.multiheadattention_layer(query, key, value)
        weight=weight.squeeze(0)
        weight=self.proj_layer(weight)
        weight = self.activation_layer(weight)

        # weight= torch.mean(weight,dim=0,keepdim=True)

        return weight

class Cluster_layer(nn.Module):

    def __init__(
        self,
        modality_name,
        cluster_number: int,
        embedding_dimension: int,
        sigma: float = 10.0,
        cluster_centers: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Module to handle the soft assignment, for a description see in 3.1.1. in Xie/Girshick/Farhadi,
        where the Student's t-distribution is used measure similarity between feature vector and each
        cluster centroid.

        :param cluster_number: number of clusters
        :param embedding_dimension: embedding dimension of feature vectors
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        :param cluster_centers: clusters centers to initialise, if None then use Xavier uniform
        """
        super(Cluster_layer, self).__init__()
        self.embedding_dimension = embedding_dimension
        self.cluster_number = cluster_number
        self.sigma = sigma
        if cluster_centers is None:
            initial_cluster_centers = torch.zeros(
                self.cluster_number, self.embedding_dimension, dtype=torch.float
            )
            nn.init.xavier_uniform_(initial_cluster_centers)
        else:
            initial_cluster_centers = cluster_centers
        self.cluster_centers = Parameter(initial_cluster_centers, requires_grad=True)
        # self.cluster_centers = initial_cluster_centers
        self.modality_name=modality_name

    def pairwise_euclidean_distance_sim(self,X: torch.Tensor, Y: torch.Tensor,
                                        eps: float = 1e-8) -> torch.Tensor:

        X_norm = (X ** 2).sum(dim=1, keepdim=True)  # (N, 1)
        Y_norm = (Y ** 2).sum(dim=1).unsqueeze(0)  # (1, M)

        XY = X @ Y.T  # (N, M)

        dist_squared = X_norm + Y_norm - 2 * XY

        dist = torch.sqrt(torch.clamp(dist_squared, min=eps))

        sim = torch.exp(- dist ** 2 / (2 * self.sigma ** 2))

        return sim

    def pairwise_pearson_tensor(self,A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:

        A_mean = A.mean(dim=1, keepdim=True)  # (7000, 1)
        B_mean = B.mean(dim=1, keepdim=True)  # (10, 1)

        A_centered = A - A_mean  # (7000, 150)
        B_centered = B - B_mean  # (10, 150)

        A_std = A_centered.norm(dim=1, keepdim=True)  # (7000, 1)
        B_std = B_centered.norm(dim=1, keepdim=True)  # (10, 1)

        cov = torch.matmul(A_centered, B_centered.T)  # (7000, 10)

        std_product = A_std @ B_std.T + 1e-8  # (7000, 10)

        corr = cov / std_product  # (7000, 10)

        return corr

    def forward(self, x):
        """
        Compute the soft assignment for a batch of feature vectors, returning a batch of assignments
        for each cluster.

        :param batch: FloatTensor of [batch size, embedding dimension]
        :return: FloatTensor [batch size, number of clusters]
        """

        if self.modality_name=='FCN' or self.modality_name=='MSN':

            return self.pairwise_pearson_tensor(x,self.cluster_centers)

        elif self.modality_name=='Pos':

            return self.pairwise_euclidean_distance_sim(x, self.cluster_centers)

        else:

            raise ValueError(f"不支持的 modality_name: '{self.modality_name}'。请检查拼写，仅支持: 'FCN', 'MSN', 'Pos'。")

class DMCP_net_autoencoder(nn.Module):

    def __init__(self,start_channels=256,in_channels=128, hidden_channels=48, out_channels=48,cluster_number=400,sigma=8.0,alpha=0.7,theta=12):
        super(DMCP_net_autoencoder, self).__init__()

        self.FC_GNNnet=GCNNet_Autoencoder(start_channels=start_channels,in_channels=in_channels, hidden_channels=hidden_channels)
        self.MSN_GNNnet=GCNNet_Autoencoder(start_channels=start_channels,in_channels=in_channels, hidden_channels=hidden_channels)

        self.FC_Cluster_layer=Cluster_layer(modality_name='FCN',cluster_number=cluster_number,embedding_dimension=hidden_channels,cluster_centers=None)
        self.MSN_Cluster_layer = Cluster_layer(modality_name='MSN', cluster_number=cluster_number,embedding_dimension=hidden_channels, cluster_centers=None)
        self.Pos_Cluster_layer = Cluster_layer(modality_name='Pos', cluster_number=cluster_number,embedding_dimension=3, sigma=sigma, cluster_centers=None)

        self.RHWM_module=Weighted_layer(cluster_center_dim=2*out_channels+3,nhead_num=3,modality_num=2)

        self.sigma=sigma
        self.alpha=alpha
        self.theta = theta

    def forward(self, data_FC_batch,data_MSN_batch):

        recon_FC,ref_group_FC=self.FC_GNNnet(data_FC_batch)
        recon_MSN,ref_group_MSN=self.MSN_GNNnet(data_MSN_batch)

        q_FC = self.FC_Cluster_layer(ref_group_FC)
        q_MSN = self.MSN_Cluster_layer(ref_group_MSN)
        q_Pos = self.Pos_Cluster_layer(data_FC_batch[0].pos)

        query=torch.concat([ref_group_FC,ref_group_MSN,data_FC_batch[0].pos],dim=-1).unsqueeze(0)
        key=value=torch.concat([self.FC_Cluster_layer.cluster_centers,self.MSN_Cluster_layer.cluster_centers,self.Pos_Cluster_layer.cluster_centers],dim=-1).unsqueeze(0)
        weight=self.RHWM_module(query,key,value)
        FC_weight = weight[:,0].unsqueeze(-1)
        MSN_weight = weight[:, 1].unsqueeze(-1)

        q=FC_weight*((1-self.alpha)*q_FC+self.alpha*q_Pos)+MSN_weight*((1-self.alpha)*q_MSN+self.alpha*q_Pos)
        q_DEC=F.softmax(q/0.1,dim=-1)
        q_onehot = F.softmax(q / 0.01, dim=-1)

        return recon_FC,recon_MSN,ref_group_FC,ref_group_MSN,weight,q_DEC,q_onehot

    def train_weight_encoder(self, loader):

        for param in self.FC_GNNnet.parameters():
            param.requires_grad = False
        for param in self.MSN_GNNnet.parameters():
            param.requires_grad = False
        for param in self.FC_Cluster_layer.parameters():
            param.requires_grad = False
        for param in self.MSN_Cluster_layer.parameters():
            param.requires_grad = False
        for param in self.Pos_Cluster_layer.parameters():
            param.requires_grad = False

        device = 'cuda:0'
        params_to_train = list(self.RHWM_module.parameters())
        optimizer = torch.optim.Adam(params_to_train, lr=0.001)
        scheduler = StepLR(optimizer, step_size=100, gamma=0.5)
        target_value = 0.5

        # Initialize the weight output to 0.5

        for epoch in range(1, 500 + 1):
            epoch_loss=0.0
            for data_MSN_batch, data_FC_batch in loader:

                data_MSN_batch = [data_MSN.to(device) for data_MSN in data_MSN_batch]
                data_FC_batch = [data_FC.to(device) for data_FC in data_FC_batch]

                optimizer.zero_grad()

                _,_,_,_,weight,_,_ = self.forward(data_FC_batch,data_MSN_batch)

                target = torch.full_like(weight, target_value)
                loss = torch.nn.functional.mse_loss(weight, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            epoch_loss /= len(loader)
            print(f"Epoch {epoch}: Loss={epoch_loss:.6f}")

        for param in self.FC_GNNnet.parameters():
            param.requires_grad = True
        for param in self.MSN_GNNnet.parameters():
            param.requires_grad = True
        for param in self.FC_Cluster_layer.parameters():
            param.requires_grad = True
        for param in self.MSN_Cluster_layer.parameters():
            param.requires_grad = True
        for param in self.Pos_Cluster_layer.parameters():
            param.requires_grad = True