import torch
import torch.nn as nn
import torch.nn.functional as F
# from model.gcn import RGCN
from torch_geometric.nn import RGCNConv
from torch_geometric.data import Data

class RGCN(nn.Module):
    def __init__(self, num_features, hidden_dim=256, output_dim=768, dropout=0.1, num_relations=3):
        super(RGCN, self).__init__()

        # 第一层 RGCN，输入维度是节点特征的维度，输出维度是 hidden_dim
        self.conv1 = RGCNConv(num_features, hidden_dim, num_relations)

        # 第二层 RGCN，输入维度是 hidden_dim，输出维度是 768
        self.conv2 = RGCNConv(hidden_dim, output_dim, num_relations)

        # Dropout 层用于防止过拟合
        self.dropout = dropout

    def forward(self, x, edge_index, edge_type):
        # 第一层图卷积
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)  # Dropout
        
        # 第二层图卷积，输出维度为 768
        x = self.conv2(x, edge_index, edge_type)

        # 返回聚合后的节点特征（输出为维度为 768 的节点嵌入）
        return x
    
#     print('Loading gcn') 
#     self.node_features, self.edge_index, self.edge_type = self.load_gcn()
#     self.gcn_proj = nn.Linear(768, self.visual_encoder.num_features)
#     self.gcn = RGCN(num_features=768, hidden_dim=384, output_dim=768, dropout=0.1, num_relations=3)
   
# def load_gcn(self):
#         node_features = []
#         edge_index = []
#         edge_type = []
#         node_features = torch.load(self.args.node_features3, weights_only=True)
#         edge_index = torch.load(self.args.edge_index3, weights_only=True)
#         edge_type = torch.load(self.args.edge_type3, weights_only=True)
#         return node_features, edge_index , edge_type

    # node_features = self.gcn_proj(node_features)
    # node_features = self.gcn(node_features, edge_index, edge_type).expand(batch_size, -1, -1)