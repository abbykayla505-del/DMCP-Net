import os
import numpy as np
from torch_geometric.data import Data
import argparse
from src.model import GCNNet_Autoencoder,DMCP_net_autoencoder
import torch
from trainer import pretrain_autoencoder,finetune_autoencoder
from scipy.io import loadmat,savemat
from scipy.stats import zscore

class PairedGraphDataset_FC(torch.utils.data.Dataset):
    def __init__(self, fc_graphs):
        self.fc_graphs = fc_graphs

    def __len__(self):
        return len(self.fc_graphs)

    def __getitem__(self, idx):
        return  self.fc_graphs[idx]

class PairedGraphDataset_MSN(torch.utils.data.Dataset):
    def __init__(self, MSN_graphs):
        self.MSN_graphs = MSN_graphs

    def __len__(self):
        return len(self.MSN_graphs)

    def __getitem__(self, idx):
        return  self.MSN_graphs[idx]

class PairedGraphDataset_FC_MSN(torch.utils.data.Dataset):
    def __init__(self, msn_graphs, fc_graphs):
        assert len(msn_graphs) == len(fc_graphs), "MSN and FC graph lists must be the same length"
        self.msn_graphs = msn_graphs
        self.fc_graphs = fc_graphs

    def __len__(self):
        return len(self.msn_graphs)

    def __getitem__(self, idx):
        return self.msn_graphs[idx], self.fc_graphs[idx]

def custom_collate_fn_FC_MSN(batch):
    msn_batch, fc_batch = zip(*batch)  # 解包成两个 list
    return list(msn_batch), list(fc_batch)

def custom_collate_fn_FC(batch):

    return list(batch)

def custom_collate_fn_MSN(batch):

    return list(batch)

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

    return dist


parser = argparse.ArgumentParser(description='Fiber clustering with DFC')
parser.add_argument(
    '--medial_wall_path', action="store", required=False,dest="medial_wall_path", default="/home/data/medial_wall_30k.mat",
    help='The file path of medial wall')
parser.add_argument(
    '--vertice_corrdinate_path', action="store", required=False,dest="vertice_corrdinate_path", default='/home/data/lh_vertice_matrix.mat',
    help='The file path of vertice corrdinates')
parser.add_argument(
    '--adj_matrix_1ring_path', action="store", required=False,dest="adj_matrix_1ring_path", default='/home/data/adj_matrix_1ring.npy',
    help='The file path of 1-ring neighborhood matrix')
parser.add_argument(
    '--neighbor_indices_arr_path', action="store", required=False,
    default='/home/data/adj_matrix_1ring_index_arr.npy',dest='neighbor_indices_arr_path',
    help='The folder path of adj_matrix_1ring_index_arr')
parser.add_argument(
    '--MSN_anchor_label_path', action="store", required=False,
    default='/home/data/Init_MSN_preparcellation_256.npy',dest="MSN_anchor_label_path",
    help='The folder path of MSN_anchor')
parser.add_argument(
    '--FC_anchor_label_path', action="store", required=False,
    default='/home/data/Init_FC_preparcellation_256.npy',dest="FC_anchor_label_path",
    help='The folder path of FC_anchor')

parser.add_argument(
    '--input_data_grouop_FC_arr_path', action="store", required=False,
    default='/home/data/FC/FC_mean_arr_30k.npy',dest="input_data_grouop_FC_arr_path",
    help='The file path of group FC matrix')
parser.add_argument(
    '--input_data_grouop_MSN_arr_path', action="store", required=False,
    default=f'/home/data/MSN/MSN_mean_arr_30k.npy',dest="input_data_grouop_MSN_arr_path",
    help='The file path of group MSN matrix')
parser.add_argument(
    '--input_data_FC_dir', action="store", required=False,
    default='/home/data/FC/FC/lh',dest="input_data_FC_dir",
    help='The folder path of individual FC data')
parser.add_argument(
    '--input_data_MSN_dir', action="store", required=False,
    default='/home/data/MSN/MSN/lh',dest="input_data_MSN_dir",
    help='The folder path of individual MSN data')
parser.add_argument(
    '--input_FC_boundary_distance_map_path', action="store", required=False,
    default='/home/data/FC_boundary_distance_map.mat',dest="input_FC_boundary_distance_map_path",
    help='The file path of FC_boundary_distance_map')
parser.add_argument(
    '--input_MSN_boundary_distance_map_path', action="store", required=False,
    default='/home/data/MSN_boundary_distance_map.mat',dest="input_MSN_boundary_distance_map_path",
    help='The file path of MSN_boundary_distance_map')
parser.add_argument(
    '--input_FC_boundary_gradient_map_path', action="store", required=False,
    default='/home/data/FC_boundary_gradient_map.mat',dest="input_FC_boundary_gradient_map_path",
    help='The file path of MSN_boundary_weight_map')
parser.add_argument(
    '--input_MSN_boundary_gradient_map_path', action="store", required=False,
    default='/home/data/MSN_boundary_gradient_map.mat',dest="input_MSN_boundary_gradient_map_path",
    help='The file path of FC_boundary_weight_map')



args = parser.parse_args()

if __name__ == '__main__':

    ''' Step 1 load data'''
    medial_wall_path = args.medial_wall_path
    vertice_corrdinate_path = args.vertice_corrdinate_path
    vertice_data = loadmat(vertice_corrdinate_path)
    vertice_data_arr = vertice_data['pos'].T
    medial_wall_data = loadmat(medial_wall_path)
    medial_wall_data_arr = medial_wall_data['medial_lh']
    index = np.where(medial_wall_data_arr.flatten() == 1)[0]
    vertice_data_arr=vertice_data_arr[index]
    cluster_neibor_path=args.adj_matrix_1ring_path
    cluster_neibor_arr=np.load(cluster_neibor_path)
    cluster_neibor_tensor=torch.from_numpy(cluster_neibor_arr)
    MSN_graphs=[]
    FC_graphs=[]
    FC_subject_dir_path = args.input_data_FC_dir
    MSN_subject_dir_path = args.input_data_MSN_dir
    FC_boundary_dist_map_path=args.input_FC_boundary_distance_map_path
    FC_boundary_dist_map = loadmat(FC_boundary_dist_map_path)
    FC_boundary_dist_map_arr = FC_boundary_dist_map['label_lh'][index]
    MSN_boundary_dist_map_path=args.input_FC_boundary_distance_map_path
    MSN_boundary_dist_map = loadmat(MSN_boundary_dist_map_path)
    MSN_boundary_dist_map_arr = MSN_boundary_dist_map['label_lh'][index]
    FC_boundary_grdient_map_path=args.input_FC_boundary_gradient_map_path
    FC_boundary_grdient_map = loadmat(FC_boundary_grdient_map_path)
    FC_boundary_grdient_map_arr = FC_boundary_grdient_map['label_lh'][index]
    MSN_boundary_grdient_map_path=args.input_FC_boundary_gradient_map_path
    MSN_boundary_grdient_map = loadmat(MSN_boundary_grdient_map_path)
    MSN_boundary_grdient_map_arr = MSN_boundary_grdient_map['label_lh'][index]
    neighbor_indices_path=args.neighbor_indices_arr_path
    neighbor_indices_arr=np.load(neighbor_indices_path)

    subject_num=0

    for subject in os.listdir(FC_subject_dir_path):

        subject_name=f"{subject.split('_')[0]}_{subject.split('_')[1]}"

        age=int(subject.split('_')[1])

        if os.path.exists(f'{MSN_subject_dir_path}/{subject_name}'):


            area_path=f'{MSN_subject_dir_path}/{subject_name}/{subject_name}_lh_area.mat'
            curv_path = f'{MSN_subject_dir_path}/{subject_name}/{subject_name}_lh_curv.mat'
            myelin_path = f'{MSN_subject_dir_path}/{subject_name}/{subject_name}_lh_myelin.mat'
            thickness_path = f'{MSN_subject_dir_path}/{subject_name}/{subject_name}_lh_thickness.mat'
            area = loadmat(area_path)
            area_data_arr = area['lh_area']
            curv = loadmat(curv_path)
            curv_data_arr = curv['lh_attri']
            myelin = loadmat(myelin_path)
            myelin_data_arr = myelin['lh_attri']
            thickness = loadmat(thickness_path)
            thickness_data_arr = thickness['lh_attri']
            MSN_feature_arr=np.concatenate((area_data_arr,curv_data_arr,myelin_data_arr,thickness_data_arr),axis=1)
            MSN_feature_arr=MSN_feature_arr[index]
            MSN_feature_arr = zscore(MSN_feature_arr, axis=0, ddof=1)
            MSN_anchor_label = np.load(args.MSN_anchor_label_path)
            MSN_anchor_label = MSN_anchor_label[index].flatten()
            MSN_arr_256 = np.concatenate(
                [np.mean(MSN_feature_arr[MSN_anchor_label == i], axis=0, keepdims=True) for i in range(0, 256)],
                axis=0)

            FC_feature_path = f'{FC_subject_dir_path}/{subject}/dtseries_lh.mat' # FC输入数据(30000,1000)
            FC = loadmat(FC_feature_path)  # 替换为你的.mat文件路径
            FC_feature_arr = FC['cifti_timecourse']
            FC_feature_arr = FC_feature_arr[index]
            FC_anchor_label = np.load(args.FC_anchor_label_path)
            FC_anchor_label = FC_anchor_label[index].flatten()
            FC_arr_256 = np.concatenate(
                [np.mean(FC_feature_arr[FC_anchor_label == i], axis=0, keepdims=True) for i in range(0, 256)],
                axis=0)

            pos_arr=vertice_data_arr

            input_data_MSN_tensor=pairwise_pearson_tensor(torch.from_numpy(MSN_feature_arr).to('cuda:0'),torch.from_numpy(MSN_arr_256).to('cuda:0')).cpu()
            input_data_FC_tensor=pairwise_pearson_tensor(torch.from_numpy(FC_feature_arr).to('cuda:0'),torch.from_numpy(FC_arr_256).to('cuda:0')).cpu()
            input_data_pos_tensor = torch.from_numpy(pos_arr)

            MSN_edges = torch.nonzero((cluster_neibor_tensor == 1), as_tuple=False).t()
            FC_edges = torch.nonzero((cluster_neibor_tensor == 1), as_tuple=False).t()

            MSN_edge_features = torch.from_numpy(
                np.load(f'{MSN_subject_dir_path}/{subject_name}/MSN_edge_features_30k.npy'))
            FC_edge_features = torch.from_numpy(np.load(f'{FC_subject_dir_path}/{subject}/FC_edge_features_30k.npy'))

            MSN_graph = Data(x=input_data_MSN_tensor.float(),pos=input_data_pos_tensor.float(), edge_index=MSN_edges,edge_attr=MSN_edge_features.float())
            FC_graph = Data(x=input_data_FC_tensor.float(),pos=input_data_pos_tensor.float(), edge_index=FC_edges,edge_attr=FC_edge_features.float())

            MSN_graphs.append(MSN_graph)
            FC_graphs.append(FC_graph)

            subject_num += 1

            print(f'complete {subject_num}th {subject_name}')


    ''' Step2 Pretrain '''

    train_MSN_AE_model_autoencoder = 1
    if train_MSN_AE_model_autoencoder==1:

        input_data_MSN_tensor=torch.from_numpy(np.load(args.input_data_grouop_MSN_arr_path))
        MSN_sim_tensor=pairwise_pearson_tensor(input_data_MSN_tensor.to('cuda:0'),input_data_MSN_tensor.to('cuda:0')).cpu()
        MSN_edges = torch.nonzero((cluster_neibor_tensor == 1), as_tuple=False).t()
        MSN_edge_features = MSN_sim_tensor[MSN_edges[0], MSN_edges[1]].unsqueeze(1)
        MSN_graph = Data(x=input_data_MSN_tensor.float(), pos=input_data_pos_tensor.float(), edge_index=MSN_edges,edge_attr=MSN_edge_features.float())
        dataset_MSN = PairedGraphDataset_MSN([MSN_graph])
        loader = torch.utils.data.DataLoader(dataset_MSN, batch_size=1, shuffle=True,collate_fn=custom_collate_fn_MSN)
        model = GCNNet_Autoencoder(start_channels=256, in_channels=128, hidden_channels=48)
        pretrain_autoencoder(model, loader, epochs=1000, log_interval=1,save_path=f"/params/pretrained_modal_MSN_autoencoder.pth", device='cuda:0')

    train_FC_AE_model_autoencoder = 1
    if train_FC_AE_model_autoencoder == 1:
        input_data_FC_tensor=torch.from_numpy(np.load(args.input_data_grouop_MSN_arr_path))
        FC_sim_tensor=pairwise_pearson_tensor(input_data_FC_tensor.to('cuda:0'),input_data_FC_tensor.to('cuda:0')).cpu()
        FC_edges = torch.nonzero((cluster_neibor_tensor == 1), as_tuple=False).t()
        FC_edge_features = FC_sim_tensor[FC_edges[0], FC_edges[1]].unsqueeze(1)
        FC_graph = Data(x=input_data_FC_tensor.float(), pos=input_data_pos_tensor.float(), edge_index=FC_edges,edge_attr=FC_edge_features.float())
        dataset_FC = PairedGraphDataset_FC([FC_graph])
        loader = torch.utils.data.DataLoader(dataset_FC, batch_size=1, shuffle=True,collate_fn=custom_collate_fn_FC)
        model = GCNNet_Autoencoder(start_channels=256, in_channels=128, hidden_channels=48)
        pretrain_autoencoder(model, loader, epochs=1000, log_interval=1,save_path=f"/params/pretrained_modal_FC_autoencoder.pth", device='cuda:0')



    ''' Step3 Finetune '''

    cluster_num=400

    dataset_FC_MSN_group = PairedGraphDataset_FC_MSN([MSN_graph], [FC_graph])
    loader_group = torch.utils.data.DataLoader(dataset_FC_MSN_group, batch_size=1, shuffle=True, collate_fn=custom_collate_fn_FC_MSN)
    dataset_FC_MSN = PairedGraphDataset_FC_MSN(MSN_graphs, FC_graphs)
    loader = torch.utils.data.DataLoader(dataset_FC_MSN, batch_size=10, shuffle=True, collate_fn=custom_collate_fn_FC_MSN)
    model=DMCP_net_autoencoder(start_channels=256,in_channels=128, hidden_channels=48, out_channels=48,cluster_number=cluster_num,sigma=8.0,alpha=0.7,theta=12)
    FC_pretrained_path=f"params/pretrained_modal_FC_autoencoder.pth"
    MSN_pretrained_path=f"params/pretrained_modal_MSN_autoencoder.pth"
    model.FC_GNNnet.load_state_dict(torch.load(FC_pretrained_path))
    model.MSN_GNNnet.load_state_dict(torch.load(MSN_pretrained_path))

    FC_boundary_dist_map_tensor=torch.from_numpy(FC_boundary_dist_map_arr)
    MSN_boundary_dist_map_tensor = torch.from_numpy(MSN_boundary_dist_map_arr)
    neighbor_indices_tensor=torch.from_numpy(neighbor_indices_arr)
    FC_boundary_grdient_map_tensor=torch.from_numpy(FC_boundary_grdient_map_arr)
    MSN_boundary_grdient_map_tensor = torch.from_numpy(MSN_boundary_grdient_map_arr)

    save_path=f"params/Fintune_modal_cluster_num_{cluster_num}_autoencoder.pth"

    finetune_autoencoder(loader_group,model, loader, epochs=200, log_interval=1,FC_boundary_dist_map=FC_boundary_dist_map_tensor,MSN_boundary_dist_map=MSN_boundary_dist_map_tensor, neighbor_indices=neighbor_indices_tensor, FC_boundary_grdient_map=FC_boundary_grdient_map_tensor,MSN_boundary_grdient_map=MSN_boundary_grdient_map_tensor,save_path=save_path)



    ''' Step3 Parcellation '''

    Jointtrain_fusion_model_path = f"params/Fintune_modal_cluster_num_{cluster_num}_autoencoder.pth"
    model.load_state_dict(torch.load(Jointtrain_fusion_model_path))
    device='cuda:0'
    model=model.to(device)
    model.eval()
    reliabel_num = 0
    label_tensor=None
    q_all=None
    with torch.no_grad():
        for data_MSN_batch_group, data_FC_batch_group in loader_group:
            data_MSN_batch_group = [data_MSN.to(device) for data_MSN in data_MSN_batch_group]
            data_FC_batch_group = [data_FC.to(device) for data_FC in data_FC_batch_group]
            recon_FC,recon_MSN,ref_group_yi_FC,ref_group_yi_MSN,_,q,q_onehot = model(data_FC_batch_group, data_MSN_batch_group)
    q=q.detach().cpu().numpy()
    label=np.argmax(q,axis=1)
    p=np.unique(label)
    label+=1
    label=label.flatten()

    final_label = np.zeros(shape=(medial_wall_data_arr.shape[0], 1))
    final_label[medial_wall_data_arr == 1] = label
    savemat(f'params/Parcellation_cluster_num_{cluster_num}_autoencoder.mat', {'label_lh': final_label})


