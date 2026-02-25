import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import numpy as np
import time
import faiss
from torch.nn import Parameter
import math
from scipy.io import loadmat, savemat
from loss import Generalized_Homogeneity_Loss,Weighted_Active_Boundary_Loss

def pretrain_autoencoder(model, loader, epochs, log_interval, save_path, device='cuda:0'):

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = StepLR(optimizer, step_size=300, gamma=0.5)
    best_loss = float('inf')
    best_model_params = None

    for epoch in range(epochs):

        epoch_loss = 0.0

        model.train()
        for data_batch in loader:
            data_batch = [data.to(device) for data in data_batch]

            optimizer.zero_grad()

            recon_yi, ref = model(data_batch)

            loss = F.l1_loss(recon_yi, data_batch[0].x, reduction='mean')

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        # adjust_learning_rate(optimizer, 0.1, epoch, epochs)

        with torch.no_grad():

            epoch_loss = epoch_loss / len(loader)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_params = model.state_dict()

            if epoch % log_interval == 0:
                print(f'Epoch [{epoch}/{epochs}],Recon_loss:{epoch_loss:.4f}')

    torch.save(best_model_params, save_path)

def finetune_autoencoder(loader_group,model, loader, epochs,log_interval,FC_boundary_dist_map,MSN_boundary_dist_map, neighbor_indices, FC_boundary_grdient_map,MSN_boundary_grdient_map,save_path, device='cuda:0'):

    def Mv_kmeans_gpu(features_MSN, features_FC, pos, n_clusters=2, max_iter_standard=300, max_iter_kmeans=300,device='cuda'):

        def kmeans_plusplus(features_MSN, features_FC, pos, n_clusters):

            selected_indices = torch.randperm(features_MSN.size(0))[:n_clusters]
            centroids_feature_MSN = features_MSN[selected_indices]
            centroids_feature_FC = features_FC[selected_indices]
            centroids_pos = pos[selected_indices]
            centroids_feature = torch.concat((centroids_feature_MSN, centroids_feature_FC), dim=1)

            return centroids_feature, centroids_pos

        def kmeans_standard(features_MSN, features_FC, pos, n_clusters=2, max_iter=300, device='cuda'):

            def pairwise_pearson_tensor(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:

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

            def pairwise_euclidean_distance_sim(X: torch.Tensor, Y: torch.Tensor,
                                                eps: float = 1e-8) -> torch.Tensor:

                X_norm = (X ** 2).sum(dim=1, keepdim=True)  # (N, 1)
                Y_norm = (Y ** 2).sum(dim=1).unsqueeze(0)  # (1, M)


                XY = X @ Y.T  # (N, M)


                dist_squared = X_norm + Y_norm - 2 * XY


                dist = torch.sqrt(torch.clamp(dist_squared, min=eps))
                sigma = model.sigma
                sim = torch.exp(- dist ** 2 / (2.0 * sigma ** 2))

                return sim


            features_MSN = features_MSN.to(device).float()  # features (B, N, D)
            features_FC = features_FC.to(device).float()  # features (B, N, D)
            features = torch.concat((features_MSN, features_FC), dim=1)
            pos = pos.to(device).float()
            centroids_feature, centroids_pos = kmeans_plusplus(features_MSN, features_FC, pos,
                                                            n_clusters)  # (B, C, D)
            # features_ex = features.unsqueeze(1).expand(-1, n_clusters, -1, -1)  # (B, N, D) -> (B, C, N, D)
            cluster_label = torch.tensor(0)
            label_matrix = torch.tensor(0)
            converged = False

            for i in range(max_iter):

                sim_pos = pairwise_euclidean_distance_sim(pos, centroids_pos)

                sim_MSN = pairwise_pearson_tensor(features_MSN, centroids_feature[:, :centroids_feature.size(1) // 2])
                q_MSN = (1-model.alpha) * sim_MSN + model.alpha * sim_pos

                sim_FC = pairwise_pearson_tensor(features_FC, centroids_feature[:, centroids_feature.size(1) // 2:])
                q_FC = (1-model.alpha) * sim_FC + model.alpha * sim_pos

                dis = 1 - (q_FC + q_MSN) / 2

                cluster_value, cluster_label = torch.min(dis, dim=1)  # (B, C, N) -> (B, N)
                cluster_value = cluster_value.unsqueeze(0)
                cluster_label = cluster_label.unsqueeze(0)

                present_clusters = torch.unique(cluster_label)

                if present_clusters.numel() < n_clusters:

                    all_clusters = torch.arange(n_clusters, device=device)  # 0, 1, …, k-1
                    isin_mask = torch.isin(all_clusters, present_clusters)
                    missing_clusters = all_clusters[~isin_mask]  # 1-D tensor
                    miss_num = missing_clusters.numel()

                    score = cluster_value

                    idx_sorted = torch.argsort(score).view(-1)  #
                    worst_idx = idx_sorted[:miss_num]

                    cluster_label[0, worst_idx] = missing_clusters.to(cluster_label.dtype)

                label_matrix = torch.zeros(cluster_label.size(0), n_clusters, cluster_label.size(-1)).to(
                    device)  # (B, C, N)
                label_matrix.scatter_(1, cluster_label.unsqueeze(1), 1)
                label_sum = torch.sum(label_matrix, dim=-1).unsqueeze(-1)  # (B, C, N) -> (B, C, 1)
                label_matrix /= label_sum
                label_matrix = label_matrix.squeeze(0).float()
                centroids_feature = label_matrix @ features  # (B, C, N)*(B, N, D)*  -> (B, C, D)
                centroids_pos = label_matrix @ pos  # (B, C, N)*(B, N, D)*  -> (B, C, D)
                min_value, _ = dis.min(dim=1, keepdim=True)
                loss_tensor = torch.sum(min_value) / min_value.size(0)
                loss = loss_tensor.item()

                if i < 10:
                    pre_label = cluster_label
                else:
                    num_diff = (cluster_label != pre_label).sum().item()

                    diff_ratio = num_diff / cluster_label.numel()
                    pre_label = cluster_label
                    if diff_ratio <= 0.01:
                        break

            return cluster_label, label_matrix, centroids_feature, centroids_pos, loss

        best_cluster_label = None
        best_label_matrix = None
        best_centroids_feature = None
        best_centroids_pos = None
        best_loss = None

        for i in range(max_iter_kmeans):
            cluster_label, label_matrix, centroids_feature, centroids_pos, loss = kmeans_standard(
                                                                                                  features_MSN,
                                                                                                  features_FC, pos,
                                                                                                  n_clusters,
                                                                                                  max_iter_standard,
                                                                                                  device)

            if i == 0:
                best_cluster_label = cluster_label
                best_label_matrix = label_matrix
                best_centroids_feature = centroids_feature
                best_centroids_pos = centroids_pos
                best_loss = loss
                print('=======================')
                print(best_loss)
                print('=======================')

            else:
                if loss < best_loss:
                    best_cluster_label = cluster_label
                    best_label_matrix = label_matrix
                    best_centroids_feature = centroids_feature
                    best_centroids_pos = centroids_pos
                    best_loss = loss
                    print('=======================')
                    print(best_loss)
                    print('=======================')

        return best_cluster_label, best_label_matrix, best_centroids_feature, best_centroids_pos, best_loss

    model = model.to(device)
    FC_boundary_dist_map = FC_boundary_dist_map.to(device)
    MSN_boundary_dist_map = MSN_boundary_dist_map.to(device)
    neighbor_indices = neighbor_indices.to(device)
    FC_boundary_grdient_map = FC_boundary_grdient_map.to(device)
    MSN_boundary_grdient_map = MSN_boundary_grdient_map.to(device)
    cluster_num = model.FC_Cluster_layer.cluster_number
    model.FC_Cluster_layer.cluster_centers = model.FC_Cluster_layer.cluster_centers.to(device)
    model.MSN_Cluster_layer.cluster_centers = model.MSN_Cluster_layer.cluster_centers.to(device)
    model.Pos_Cluster_layer.cluster_centers = model.Pos_Cluster_layer.cluster_centers.to(device)

    model.eval()
    ref_FC_mean=None
    ref_MSN_mean=None
    with torch.no_grad():
        for data_MSN_batch, data_FC_batch in loader_group:

            data_MSN_batch = [data_MSN.to(device) for data_MSN in data_MSN_batch]
            data_FC_batch = [data_FC.to(device) for data_FC in data_FC_batch]

            _,_,ref_group_yi_FC,ref_group_yi_MSN,_,_,_ = model(data_FC_batch,data_MSN_batch)

            if ref_FC_mean is not None:
                ref_FC_mean += ref_group_yi_FC.detach()
            else:
                ref_FC_mean = ref_group_yi_FC.detach()
            if ref_MSN_mean is not None:
                ref_MSN_mean += ref_group_yi_MSN.detach()
            else:
                ref_MSN_mean = ref_group_yi_MSN.detach()

    ref_FC_mean = ref_FC_mean / len(loader)
    ref_MSN_mean = ref_MSN_mean / len(loader)
    ref_Pos_mean = data_FC_batch[0].pos

    start_time = time.time()
    best_cluster_label, best_label_matrix, best_centroids_feature, best_centroids_pos, best_loss = Mv_kmeans_gpu(ref_MSN_mean,
                                                                                                              ref_FC_mean,
                                                                                                              data_MSN_batch[
                                                                                                                  0].pos,
                                                                                                              n_clusters=cluster_num,
                                                                                                              max_iter_standard=1000,
                                                                                                              max_iter_kmeans=50,
                                                                                                              device=device)
    end_time = time.time()
    print(f"Cost time: {end_time - start_time:.4f}s")

    label = best_cluster_label.cpu().numpy().flatten()

    FC_feature=ref_FC_mean.detach().cpu().numpy()
    MSN_feature = ref_MSN_mean.detach().cpu().numpy()
    Pos_feature = ref_Pos_mean.detach().cpu().numpy()
    Init_FC_center_np = np.array([np.mean(FC_feature[label == k], axis=0) for k in range(cluster_num)])
    Init_MSN_center_np = np.array([np.mean(MSN_feature[label == k], axis=0) for k in range(cluster_num)])
    Init_Pos_center_np = np.array([np.mean(Pos_feature[label == k], axis=0) for k in range(cluster_num)])
    Init_FC_center=torch.from_numpy(Init_FC_center_np).to(device)
    Init_MSN_center = torch.from_numpy(Init_MSN_center_np).to(device)
    Init_Pos_center = torch.from_numpy(Init_Pos_center_np).to(device)

    model.FC_Cluster_layer.cluster_centers=Parameter(Init_FC_center, requires_grad=True)
    model.MSN_Cluster_layer.cluster_centers = Parameter(Init_MSN_center, requires_grad=True)
    model.Pos_Cluster_layer.cluster_centers = Parameter(Init_Pos_center, requires_grad=True)

    model.train_weight_encoder(loader_group)

    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=0.01)
    scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

    best_loss = float('inf')
    best_model_params = None
    homogeneity_Loss = Generalized_Homogeneity_Loss(sigma=model.sigma, alpha=model.alpha)
    graphABL_Loss= Weighted_Active_Boundary_Loss(num_active_points=400,max_clip_dist=model.theta)
    batch_num = 0

    for epoch in range(1, epochs + 1):

        model.train()

        epoch_loss = 0.0
        epoch_loss_contras = 0.0
        epoch_loss_homo = 0.0
        epoch_loss_boundary = 0.0
        FC_homo = 0.0
        MSN_homo = 0.0


        for data_MSN_batch, data_FC_batch in loader:

            data_MSN_batch = [data_MSN.to(device) for data_MSN in data_MSN_batch]
            data_FC_batch = [data_FC.to(device) for data_FC in data_FC_batch]

            optimizer.zero_grad()

            for data_MSN_batch_group, data_FC_batch_group in loader_group:

                recon_FC,recon_MSN,ref_group_yi_FC,ref_group_yi_MSN,_,q,q_onehot = model(data_FC_batch_group, data_MSN_batch_group)


            # Recon loss
            recon1 = 0.5*F.l1_loss(recon_FC, data_FC_batch_group[0].x, reduction='mean')+0.5*F.l1_loss(recon_MSN, data_MSN_batch_group[0].x, reduction='mean')

            # Homo loss
            homo_losses = []
            loss2_homo_FC = 0.0
            loss2_homo_MSN = 0.0

            for i in range(len(data_FC_batch)):
                loss_val, metric_val1 = homogeneity_Loss(data_FC_batch[i].x, q_onehot, data_FC_batch[i].pos)
                loss_val2, metric_val2 = homogeneity_Loss(data_MSN_batch[i].x, q_onehot, data_MSN_batch[i].pos)

                loss_homo = (0.5 * loss_val + 0.5 * loss_val2)

                homo_losses.append(loss_homo)
                loss2_homo_FC += metric_val1
                loss2_homo_MSN += metric_val2

            homogeneityloss2 = torch.sum(torch.stack(homo_losses)) / len(data_FC_batch)
            loss2_homo_FC_mean = loss2_homo_FC / len(data_FC_batch)
            loss2_homo_MSN_mean = loss2_homo_MSN / len(data_MSN_batch)

            # Boundary loss
            boundaryloss3=0.5*graphABL_Loss(q, FC_boundary_dist_map, neighbor_indices, FC_boundary_grdient_map)+0.5*graphABL_Loss(q, MSN_boundary_dist_map, neighbor_indices, MSN_boundary_grdient_map)

            # Ours
            loss = recon1+homogeneityloss2+epoch/epochs*boundaryloss3

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_loss_contras+=recon1.item()
            epoch_loss_homo +=homogeneityloss2.item()
            epoch_loss_boundary +=boundaryloss3.item()
            FC_homo +=loss2_homo_FC_mean
            MSN_homo +=loss2_homo_MSN_mean

            batch_num+=1

        scheduler.step()

        with torch.no_grad():

            epoch_loss = epoch_loss / len(loader)
            epoch_loss_contras=epoch_loss_contras/ len(loader)
            epoch_loss_homo=epoch_loss_homo/ len(loader)
            epoch_loss_boundary=epoch_loss_boundary / len(loader)
            FC_homo=FC_homo / len(loader)
            MSN_homo=MSN_homo / len(loader)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_params = model.state_dict()

            if epoch % log_interval == 0:
                print(f'Epoch [{epoch}/{epochs}]: Joint_loss:{epoch_loss:.4f}; Recon_loss:{epoch_loss_contras:.4f}; Homogenity_loss:{epoch_loss_homo:.4f}; Boundary_loss:{epoch_loss_boundary:.4f}; FC_homo:{FC_homo:.4f}; MSN_homo:{MSN_homo:.4f};')


    # Save the best model and (optionally) the cluster centers.
    torch.save(best_model_params, save_path)