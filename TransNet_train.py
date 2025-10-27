import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np
from torch_geometric.utils import dense_to_sparse, dropout_adj
from torch_scatter import scatter_add
from scipy.spatial import distance
from geopy.distance import geodesic
import metpy.calc as mpcalc
from metpy.units import units
from datetime import datetime, timedelta
from graph import graph 
from tqdm import tqdm
import pandas as pd
# from pytorch_tcn import TCN
# from torch.sparse.linalg import cg
from scipy.sparse.linalg import cg
from torch_geometric.nn import ChebConv


def setup(rank, world_size):
    """
    Initialize the distributed environment
    """
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ["MASTER_PORT"] = '12355'
    dist.init_process_group("nccl", rank = rank, world_size = world_size)
    
def cleanup():
    """
    Clean up distributed environment
    """
    dist.destroy_process_group()
    
class DirectionalWeights(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        """
        Implements Algorithm 2 for learning directional edge weights.
        A1, A2, A3, A4 are implemented as fully connected layers as specified in the paper.
        
        Args:
            feature_dim: Dimension of input node features
            hidden_dim: Dimension of hidden representations
        """
        super(DirectionalWeights, self).__init__()
        
        # Convert A1, A2, A3, A4 to fully connected layers
        self.A1 = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                nn.LayerNorm(hidden_dim),
                                # nn.Linear(hidden_dim * 2, hidden_dim),
                                # nn.LayerNorm(hidden_dim),
                                # nn.ReLU(), 
                               )
        self.A2 = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                nn.LayerNorm(hidden_dim),
                                # nn.Linear(hidden_dim * 2, hidden_dim),
                                # nn.LayerNorm(hidden_dim),
                                # nn.ReLU(),
                               )
        self.A3 = nn.Linear(hidden_dim, 1)   #$ nn.Sequential(
                               # nn.LayerNorm(1),
                               # )
        self.A4 = nn.Linear(1, 1)
        self.scaling_factor = nn.Parameter(torch.ones(1))
        
    def forward(self, node_features, edge_index, num_nodes):
        batch_size = node_features.shape[0]
        num_edges = edge_index.shape[-1]
        
        Vij_normalized = torch.zeros(batch_size, num_edges, device=node_features.device)
        Vji_normalized = torch.zeros(batch_size, num_edges, device=node_features.device)
        
        for b in range(batch_size):
            # Get source and target indices for this batch
            src_nodes = edge_index[b, 0]  # (num_edges,)
            dst_nodes = edge_index[b, 1]  # (num_edges,)
            
            # Get source and target node features
            Ui = node_features[b, src_nodes]  # (num_edges, feature_dim)
            Uj = node_features[b, dst_nodes]  # (num_edges, feature_dim)
            
            # Compute Zij and Zjitorch.tanh
            Zij = F.relu(self.A1(Ui) + self.A2(Uj))  # (num_edges, hidden_dim), F.relu  
            Zij = self.A3(Zij)  # (num_edges, 1)
            
            # For Zji, we use the same transformations but swap i and j
            Zji = F.relu(self.A1(Uj) + self.A2(Ui))  # (num_edges, hidden_dim), F.relu  
            Zji = self.A3(Zji)  # (num_edges, 1)
            
            # Compute the difference terms
            diff_ij = self.A4(Zij - Zji)  # This will be opposite of ji direction
            diff_ji = self.A4(-Zij + Zji)
            
            # Apply ReLU to ensure one direction is zero while other is non-negative
            Vij = F.relu(diff_ij).squeeze(-1)  # (num_edges,)
            Vji = F.relu(diff_ji).squeeze(-1)  # (num_edges,)
            
            # Vij = Vij / (torch.sum(Vij, dim=-1, keepdim=True) + 1e-6)
            # Vji = Vji / (torch.sum(Vji, dim=-1, keepdim=True) + 1e-6)
            # Normalize weights per source node using softmax
            for i in range(num_nodes):
                mask_i = (src_nodes == i)
                if mask_i.any():
                    Vij_normalized[b, mask_i] = F.softmax(Vij[mask_i], dim=0)
            
            for j in range(num_nodes):
                mask_j = (dst_nodes == j)
                if mask_j.any():
                    Vji_normalized[b, mask_j] = F.softmax(Vji[mask_j], dim=0)
        
        return Vij_normalized, Vji_normalized

class GraphProcessor:
    """Process spatial data to create graph structure with support for batched input"""
    def __init__(self, lats, lons, dist_thres=0.4):
        """
        Initialize the graph processor
        Args:
            lats: Latitude coordinates of nodes
            lons: Longitude coordinates of nodes
            dist_thres: Distance threshold for connecting nodes
        """
        self.lats = lats
        self.lons = lons
        self.node_num = len(lats)
        self.dist_thres = 0.6 # dist_thres
        
        # Pre-compute static spatial components
        self.coords = np.stack([self.lons, self.lats], axis=1)
        self.dist_matrix = distance.cdist(self.coords, self.coords, 'euclidean')
        
        # Create static adjacency matrix
        self.adj = np.zeros((self.node_num, self.node_num), dtype=np.uint8)
        self.adj[self.dist_matrix <= self.dist_thres] = 1
        np.fill_diagonal(self.adj, 0)
        
        # Pre-compute static edge information
        self.edge_index, _ = dense_to_sparse(torch.tensor(self.adj))
        
        # Pre-compute edge spatial features
        src_coords = self.coords[self.edge_index[0]]
        dst_coords = self.coords[self.edge_index[1]]
        
        # Calculate distances and direction between nodes (stations)
        self.city_distances = torch.tensor(
            np.sqrt(np.sum((dst_coords - src_coords)**2, axis=1)),
            dtype = torch.float32
        )
        
        self.city_directions = torch.tensor(
            np.arctan2(
                dst_coords[:, 1] - src_coords[:, 1],
                dst_coords[:, 0] - src_coords[:, 0]
            ),
            dtype = torch.float32
        )
        self.edge_attr = torch.stack([self.city_distances, self.city_directions], dim=1)
        self.edge_attr_norm = (self.edge_attr - self.edge_attr.mean(dim = 0)) / self.edge_attr.std(dim = 0)
        
    def construct_graph(self, wind_data):
        """
        Construct graphs from batched data samples
        Args:
            wind_batch: Shape (batch_size, 2, n_stations) containing wind speed and wind direction # torch.Size([32, 2, 170])
        Returns:
            edge_index: Shape (batch_size, 2, num_edges)
            edge_weight: Shape (batch_size, num_edges)
            edge_feature: Shape (batch_size, num_edges, feature_dim) including edge weights
        """
        batch_size = wind_data.shape[0]
        device = wind_data.device
        
        # Convert static tensors to device
        edge_index = self.edge_index.to(device)
        city_distances = self.city_distances.to(device)
        city_directions = self.city_directions.to(device)
        edge_attr_norm = self.edge_attr_norm.to(device)   # _norm
        
        # Prepare batched edge indices
        batch_edge_index = edge_index.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Extract wind features for source node
        src_wind_speeds = torch.stack([
            wind_data[b, 0, edge_index[0]] 
            for b in range(batch_size)
        ])
        
        src_wind_directions = torch.stack([
            wind_data[b, 1, edge_index[0]]
            for b in range(batch_size)
        ])
        
        # Calculate angle differences
        # print(f'Shape of city direction is {city_directions.shape}')
        # print(f'Shape of city direction is {src_wind_directions.shape}')
        theta = torch.abs(city_directions.unsqueeze(0) - src_wind_directions)
        edge_weights = F.relu(src_wind_speeds * torch.cos(theta) / city_distances.unsqueeze(0))
        
        # Creating edge features 
        edge_attr_batch = edge_attr_norm.unsqueeze(0).expand(batch_size, -1, -1)
        edge_features = torch.cat([
            edge_attr_batch,
            edge_weights.unsqueeze(-1)
        ], dim = -1)
        # print(batch_edge_index.shape)
        # print(batch_edge_index)
        # # torch.save(batch_endge_index,)
        # asd
        return batch_edge_index, edge_weights, edge_features   # torch.Size([2, 7932]) and torch.Size([32, 7932])
        

class TransNet(nn.Module):
    def __init__(self, input_features, n_stations, seq_length, pred_length, L, state_dim, hist_dim):
        super(TransNet, self).__init__()
        
        # Model dimensions
        self.input_features = input_features
        self.n_stations = n_stations
        self.L = L
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.device = None
        self.h = 0.01
        
        #################################################### Input Transformation ####################################################
        self.pollutants = ['PM25_mode_0', 'PM25_mode_1', 'PM25_mode_2', 'PM25_mode_3', 'PM25_mode_4', 'PM25_mode_5', 'pollutant_pca_0', 'pollutant_pca_1', 'pollutant_pca_2', 'PM25']
        self.pollutant_indices = [33, 34, 35, 36, 37, 38, 39, 40, 41, 42]
        self.pollutant_size = len(self.pollutants)
        # self.pollutant_state = nn.Sequential(
        #     nn.Linear(self.pollutant_size, self.pollutant_size),
        #     nn.LayerNorm(self.pollutant_size),
        #     # nn.ReLU(),
        # )
        
        self.pca = ['pollutant_pca_0', 'pollutant_pca_1', 'pollutant_pca_2']
        self.pca_indices = [39, 40, 41]
        self.pca_size = len(self.pca)
        
        self.wind_features = ['U_WIND', 'V_WIND', 'WDIR10_cos', 'WDIR10_sin']
        self.wind_indices = [0, 1, 2, 3]
        self.wind_size = 24 * len(self.wind_features)
        self.wind_emb = self.wind_size
        # self.wind_hist = nn.Sequential(
        #     nn.Linear(self.wind_size, self.wind_emb),
        #     nn.LayerNorm(self.wind_emb),
        #     # nn.ReLU(),
        # )

        self.temporal_features = ['hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                                'weekday_sin', 'weekday_cos', 'dayofyear_sin', 'dayofyear_cos']
        self.temporal_indices = [4, 5, 6, 7, 8, 9, 10, 11]
        self.temporal_size = len(self.temporal_features)
        # self.temporal_state = nn.Sequential(
        #     nn.Linear(self.temporal_size, self.temporal_size)
        # )

        self.pos_features = ['pe_0', 'pe_1', 'pe_2', 'pe_3', 'pe_4', 'pe_5', 'pe_6', 'pe_7', 'pe_8', 'pe_9', 'pe_10', 'pe_11', 'pe_12', 'pe_13', 'pe_14', 'pe_15']
        self.pos_indices = [12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]
        self.pos_size = len(self.pos_features)
        # self.pos_state = nn.Sequential(
        #     nn.Linear(self.pos_size, self.pos_size),
        #     # nn.Linear(self.pos_size, self.pos_size),
        # )

        
        self.met_features = ['PBL', 'TEMP2', 'Q2', 'RN', 'RC']
        self.met_indices = [28, 29, 30, 31, 32]
        self.met_size = len(self.met_features) * 24
        # self.met_state = nn.Sequential(
        #     nn.Linear(self.met_size, self.met_size),
        #     nn.LayerNorm(self.met_size),
        #     nn.ReLU(),
        # )

        # self.feature_size = 43 #self.pollutant_size + len(self.wind_features) + self.temporal_size + self.pos_size + self.met_size
        # self.x_temporal_emb = nn.Sequential(
        #     nn.Linear(self.feature_size, self.feature_size),
        #     nn.LayerNorm(self.feature_size),
        #     # nn.ReLU(self.feature_size),
        # )
        
        # Define latent dimensions for embeddings
        self.time_input = 24 * self.temporal_size
        self.time_emb = self.time_input
        self.g_time = nn.Sequential(
            nn.Linear(self.time_input, self.time_emb),
            # nn.Linear(self.time_emb, self.time_emb),cd 
            )
        #  + self.met_size  + self.met_size  + self.wind_size +
        
        self.state_input_size = self.pollutant_size + (24 * self.pos_size) + self.time_emb
        self.state_latent_size = self.state_input_size
        self.g_state_PM = nn.Sequential(
            nn.Linear(self.state_input_size, self.state_latent_size),
            nn.LayerNorm(self.state_latent_size),
            # nn.LeakyReLU(),
            # nn.Linear(self.state_input_size, self.state_latent_size),
            # nn.LayerNorm(self.state_latent_size),
            # nn.Tanh(),
            )

        self.hist_input_size = 24 * (self.pos_size) + self.time_emb + self.wind_emb
        self.hist_latent_size = self.hist_input_size                  
        self.g_hist_wind = nn.Sequential(
            nn.Linear(self.hist_input_size, self.hist_latent_size),
            nn.LayerNorm(self.hist_latent_size),
            # nn.LeakyReLU(),
            # nn.Linear(self.hist_input_size, self.hist_latent_size),
            # nn.LayerNorm(self.hist_latent_size),
            # nn.Tanh(),
            )
        # self.cheb_conv = ChebConv(self.state_latent_size, self.state_latent_size, K=3)
        #################################################################################################################################
        
        #################################################### Directional weight for advection direction ####################################################
        self.directional_weights = DirectionalWeights(self.hist_latent_size, self.hist_latent_size)
        
        self.diffusion_params = nn.Parameter(torch.randn(self.hist_latent_size))
        
        self.reaction_networks = nn.ModuleDict({
                'R1': nn.Sequential(
                    nn.Linear(self.state_latent_size, self.state_latent_size),
                    nn.LayerNorm(self.state_latent_size),
                    # nn.Linear(self.state_latent_size * 2, self.state_latent_size),
                    # nn.LayerNorm(self.state_latent_size),
                    # nn.ReLU()
                    ),
                'R2': nn.Sequential(
                    nn.Linear(self.state_latent_size, self.state_latent_size),
                    nn.LayerNorm(self.state_latent_size),
                    # nn.Linear(self.state_latent_size * 2, self.state_latent_size),
                    # nn.LayerNorm(self.state_latent_size),
                    # nn.ReLU()
                    ),
                'R3': nn.Sequential(
                    nn.Linear(self.pollutant_size * 24, self.state_latent_size),
                    nn.LayerNorm(self.state_latent_size),
                    # nn.Linear(self.state_latent_size * 2, self.state_latent_size),
                    # nn.LayerNorm(self.state_latent_size),
                    # nn.ReLU(),
                    ),
                'R4': nn.Sequential( # + self.time_emb + (self.pos_size * 24)
                    nn.Linear(self.met_size , self.state_latent_size),
                    nn.LayerNorm(self.state_latent_size),
                    # nn.Linear(self.state_latent_size * 2, self.state_latent_size),
                    # nn.LayerNorm(self.state_latent_size),
                    # nn.ReLU(),
                    ),
            })

        combined_feature_size = self.state_latent_size # + self.time_emb + (self.pos_size * 24)
        self.g_out = nn.Sequential(
            nn.Linear(combined_feature_size, combined_feature_size // 2),
            nn.LayerNorm(combined_feature_size // 2),
            # nn.ReLU(),
            # nn.Tanh(),
            nn.Linear(combined_feature_size // 2, 72),
            nn.LayerNorm(72),
            # nn.ReLU(),
            # nn.Linear(self.state_latent_size // 4, 72),
            # nn.ReLU()
        )
        
        self.graph_processor = None
    
    def _ensure_float(self, tensor):
        """Ensure tensor is float and on the correct device."""
        return tensor.float().to(self.device)
        
    def initialize_graph_processor(self, lats, lons):
        self.graph_processor = GraphProcessor(lats, lons)
    
    def to(self, device):
        """Override to method to capture device"""
        self.device = device
        return super().to(device)
    
    def get_K(self, x):
        return F.hardtanh(x, min_val = 0.0, max_val = 1.0)
        
    def compute_advection_term(self, u_state, u_hist, edge_index, edge_weight, G, h):
        """
        Compute advection using both physics-based and learned directional weights
    
        Args:
            u_state: Node features (batch_size, hidden_dim, n_stations) torch.Size([32, 1024, 170])   
            u_hist: History features (batch_size, hidden_dim, n_stations) torch.Size([32, 1024, 170])
            edge_index: Edge indices (batch_size, 2, num_edges) torch.Size([32, 2, 7932])
            edge_weights: Initial weights from wind parameters (batch_size, num_edges) torch.Size([32, 7932])
            G: Graph instance
    
        Returns:
            Advection term (batch_size, hidden_dim, n_stations)
        """
        # Ensure consistent tensor type
        u_state = u_state.float()  # Convert to float
        u_hist = u_hist.float()    # Ensure float type
        edge_index = edge_index.long()  # Ensure long type for indexing
        
        Vij, Vji = self.directional_weights(u_hist.transpose(1, 2), edge_index, self.n_stations)
        if len(Vji.shape) > 2:
            Vji = Vji.squeeze()
        Vji = Vji.float()# * edge_weight.float()
        weighted_neighbors = G.neighborNode(u_state, W=Vji)          
        
        aggregated_neighbors = torch.zeros_like(u_state, dtype=torch.float32)
        for b in range(u_state.shape[0]):
            aggregated_neighbors[b].index_add_(1, edge_index[b,1], weighted_neighbors[b].to(aggregated_neighbors.dtype))
        
        current_features = u_state
        advection = (aggregated_neighbors - current_features)
        return (u_state + h * advection), Vji
    
    def compute_diffusion_term(self, u_state, edge_index, edge_weight, G, K, h):
        """
        Compute diffusion following the paper's equation:
        U(l+2/3) = mat((I + hK(l) ⊗ L)^(-1)vec(U(l+1/3)))
    
        Args:
            u_state: Node features (batch_size, hidden_dim, n_stations)
            edge_index: Edge indices (2, num_edges)
            edge_weight: Edge weights (batch_size, num_edges)
            G: Graph instance
            K: Feature-wise diffusion coefficients (channels)
        
        Returns:
            Diffusion term (batch_size, hidden_dim, n_stations)
        """
        u_state = u_state.float()
        edge_index = edge_index.long()
        edge_weight = edge_weight.float()
        K = K.float()
        
        batch_size, hidden_dim, n_stations = u_state.shape
    
        identity = torch.eye(n_stations, device=u_state.device)
        diffused = torch.zeros_like(u_state)
        for b in range(batch_size):
            G_b = graph(edge_index[b, 0], edge_index[b, 1], self.n_stations, edge_weight[b], device=u_state.device)
            L_b = G_b.get_normalized_laplacian()
            L_b = L_b.to(u_state.device).float()
            for c in range(hidden_dim):
                kc = K[c].item()
                M = identity + h * kc * L_b + 1e-6 * identity
                
                try:
                    diffused[b, c] = torch.linalg.solve(M, u_state[b, c])
                except RuntimeError:
                    result = torch.linalg.lstsq(M, u_state[b, c].unsqueeze(1))
                    solution = result.solution
                    diffused[b, c] = solution.squeeze()
                except Exception as e:
                    M_inv = torch.linalg.pinv(M, rcond=1e-6)
                    diffused[b, c] = torch.matmul(M_inv, u_state[b, c])
        
        return diffused

    def compute_reaction_term(self, state, hist, g_time_input, met_hist, pos_hist, R1, R2, R3, R4, h):
        """
        Compute reaction following the paper's equation:
        U(l+1) = U(l+2/3) + hf(U(l+2/3), U_hist(0))
    
        Args:
            u_state: Current state (batch_size, n_stations, hidden_dim)
            u_hist: History state (batch_size, n_stations, hidden_dim)
        
        Returns:
            Updated state with reaction (batch_size, n_stations, hidden_dim)
        """
        r1_output = R1(state)
        r2_output = torch.tanh(R2(state)) * (state)
        r3 = R3(hist)
        met = met_hist # torch.cat([met_hist, g_time_input, pos_hist], dim = -1)
        r4 = R4(met)
        final_output = F.relu(r1_output + r2_output + r3 + r4)
        return (h * final_output) + state

    def recover_wind_parameters(self, u_wind, v_wind, wdir_cos=None, wdir_sin=None):
        """
        Recover wind speed and direction from decomposed components
        
        Args:
            u_wind: U component of wind (zonal)
            v_wind: V component of wind (meridional)
            wdir_cos: Cosine component of wind direction (optional)
            wdir_sin: Sine component of wind direction (optional)
        
        Returns:
            wind_speed: Magnitude of wind vector
            wind_direction: Direction of wind in degrees (0-360, meteorological convention)
        """
        wind_speed = torch.sqrt(u_wind**2 + v_wind**2)
        if wdir_cos is None or wdir_sin is None:
            wind_direction = 270 - torch.atan2(v_wind, u_wind) * 180 / torch.pi
            wind_direction = wind_direction #% 360
        else:
            wind_direction = torch.atan2(wdir_sin, wdir_cos) * 180 / torch.pi
            wind_direction = wind_direction #% 360
    
        return wind_speed, wind_direction
    
    
    def forward(self, x_temporal):
        """
        Args:
            x_temporal: Shape (batch, seq_length = 24, n_stations = 170, features = 8)
            t_emb: Shape (batch, seq_length = 24, n_stations = 170, features = 10)
        """
        x_temporal = self._ensure_float(x_temporal)
        # x_temporal = F.dropout(x_temporal, p=0.1, training=self.training)
        # x_temporal = self.x_temporal_emb(x_temporal)
        current_h = self.h
        batch_size = x_temporal.size(0)

        pollutants = x_temporal[..., self.pollutant_indices]
        pollutants_hist = pollutants.permute(0, 2, 1, 3).reshape(batch_size, self.n_stations, -1)
        
        wind = x_temporal[..., self.wind_indices]
        wind_hist = wind.permute(0, 2, 1, 3).reshape(batch_size, self.n_stations, -1)
        # wind_hist = self.wind_hist(wind_hist)
        
        wind_speed, wind_direction = self.recover_wind_parameters(wind[..., 0], wind[..., 1], wind[..., 2], wind[..., 3])
        wind_data = torch.stack([wind_speed, wind_direction], dim=1)
        
        temporal = x_temporal[..., self.temporal_indices]
        g_time_input = temporal.permute(0, 2, 1, 3).reshape(batch_size, self.n_stations, -1)
        g_time_input = self.g_time(g_time_input)
        
        pos = x_temporal[..., self.pos_indices]
        # pos = self.pos_state(pos)
        pos_hist = pos.permute(0, 2, 1, 3).reshape(batch_size, self.n_stations, -1)
        
        met = x_temporal[..., self.met_indices]
        met_hist = met.permute(0, 2, 1, 3).reshape(batch_size, self.n_stations, -1)
        
        g_state_PM = torch.cat([
            pollutants[:, -1], pos_hist, g_time_input,
        ], dim = -1)
        g_state_PM = F.dropout(g_state_PM, p=0.1, training=self.training)
        g_state_PM = self.g_state_PM(g_state_PM)
        
        # pollutants_hist = pollutants.permute(0, 2, 1, 3).reshape(batch_size, self.n_stations, -1)
        
        
        # print(wind_hist.shape, pos_hist.shape, g_time_input.shape)
        g_hist_wind = torch.cat([
            wind_hist, pos_hist, g_time_input
        ], dim = -1)
        g_hist_wind = F.dropout(g_hist_wind, p=0.3, training=self.training)
        g_hist_wind = self.g_hist_wind(g_hist_wind)
        
        device = x_temporal.device
        edge_index, edge_weights, _ = self.graph_processor.construct_graph(wind_data[:, :, -1, :])
        edge_index = edge_index.long()
        edge_weights = edge_weights.float()
        G = graph(edge_index[:, 0], edge_index[:, 1], self.n_stations, edge_weights, device=self.device)
        # edge_src, edge_target = edge_index[0, 0].long(), edge_index[0, 1].long()
        
        # Algorithm 4 Lines 6-12: ADR iterations
        for l in range(self.L):
            # g_state_PM = F.dropout(g_state, p=0.3, training=self.training)
            g_state_PM = g_state_PM.transpose(1, 2)
            state_advection, _ = self.compute_advection_term(g_state_PM, g_hist_wind.transpose(1, 2), edge_index, edge_weights, G, h = current_h) 
            
            K = self.get_K(self.diffusion_params)
            state_diffusion = self.compute_diffusion_term(state_advection, edge_index, edge_weights, G, K, h = current_h)
            state_diffusion = state_diffusion.transpose(1, 2)
            state_reaction = self.compute_reaction_term(
                state_diffusion, pollutants_hist, g_time_input, met_hist, pos_hist,
                self.reaction_networks['R1'], self.reaction_networks['R2'], self.reaction_networks['R3'], self.reaction_networks['R4'],
                h = current_h)
        
        next_state = F.dropout(state_reaction, p=0.1, training=self.training)
        # final_state = torch.cat([
        #     next_state, pos_hist, g_time_input,
        # ], dim = -1)
        next_state = self.g_out(next_state)
        return next_state.permute(0, 2, 1)
        
class custom_loss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, gamma=1.0, delta=0.1):
        """
        Custom loss function combining MAE, MSE, IOA, and temporal smoothness.

        Args:
            alpha (float): Weight for MAE loss.
            beta (float): Weight for MSE loss.
            gamma (float): Weight for IOA loss (negative sign to maximize IOA).
            delta (float): Weight for temporal smoothness penalty.
        """
        super(custom_loss, self).__init__()
        self.mae_loss = nn.L1Loss()

    def forward(self, predicted, target):
        """
        Compute the loss.

        Args:
            predicted (torch.Tensor): Predicted PM2.5 values (batch, time, nodes).
            target (torch.Tensor): Ground truth PM2.5 values (batch, time, nodes).

        Returns:
            torch.Tensor: Computed loss value.
        """
        mae = self.mae_loss(predicted, target)
        # mse = self.mse_loss(predicted, target)

        # Compute IOA (Index of Agreement)
        epsilon = 1e-6
        target_mean = target.mean(dim=-1, keepdim=True)
        numerator = ((predicted - target) ** 2).sum(dim=-1)
        denominator = ((torch.abs(predicted - target_mean) + 
                       torch.abs(target - target_mean)) ** 2).sum(dim=-1) + epsilon
        ioa = 1 - (numerator / denominator).mean()
        return mae, -ioa
        