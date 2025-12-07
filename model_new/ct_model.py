import math
from typing import Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (batch_first: True)."""
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # [max_len, d_model]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence_length = inputs.size(1)
        return inputs + self.pe[:sequence_length].unsqueeze(0)


class CTSequenceModel(nn.Module):
    """
    Continuous-time Mw(t) prediction model with Transformer encoder, 
    station attention/weighting, status soft/hard gating, and optional heteroscedastic output.

    Input:  x in shape [batch, time_steps, num_stations * 2] where last dim packs (PGD, status).
    Output: 
      - if heteroscedastic=True: (mu, log_var) both in shape [batch, time_steps]
      - else: y in shape [batch, time_steps]
    """
    def __init__(
        self,
        num_stations: int,
        features_per_station: int = 2,
        model_dim: int = 128,
        num_attention_heads: int = 4,
        num_encoder_layers: int = 2,
        feedforward_dim: int = 256,
        dropout_rate: float = 0.2,
        post_hidden_units: int = 128,
        input_dropout_rate: float = 0.0,
        enable_station_attention: bool = True,
        attention_hidden_units: int = 16,
        normalize_station_attention: bool = False,
        attention_temperature: float = 1.0,
        enable_hard_mask: bool = False,
        enable_heteroscedastic_output: bool = True,
        enable_causal_mask: bool = False,
        enable_geo_features: bool = False,
        geo_feature_mode: str = "xyz",
        geo_embedding_dim: int = 16,
        pgd_value_clip: float | None = None,
    ):
        super().__init__()
        self.num_stations = num_stations
        self.feature_dim = features_per_station
        self.d_model = model_dim
        self.hard_mask = enable_hard_mask
        self.use_station_attention = enable_station_attention
        self.normalize_station_attention = normalize_station_attention
        self.attn_temperature = float(attention_temperature)
        self.heteroscedastic = enable_heteroscedastic_output
        self.use_causal_mask = enable_causal_mask
        self.use_geo = enable_geo_features
        self.geo_mode = geo_feature_mode
        self.geo_embed_dim = geo_embedding_dim
        self.pgd_clip = pgd_value_clip
        # runtime buffer for per-station geo embedding [N, Dg]
        self.station_geo_embed: Optional[torch.Tensor] = None

        # 改进的地理编码器：基于台站位置的多层特征提取
        if self.use_geo:
            if self.geo_mode == 'xyz':
                self.geo_in_dim = 3
            elif self.geo_mode == 'sin_cos':
                self.geo_in_dim = 4
            else:
                self.geo_in_dim = 2
            self.geo_encoder = nn.Sequential(
                nn.Linear(self.geo_in_dim, geo_embedding_dim // 2),
                nn.GELU(),
                nn.LayerNorm(geo_embedding_dim // 2),
                nn.Linear(geo_embedding_dim // 2, geo_embedding_dim),
                nn.GELU(),
                nn.LayerNorm(geo_embedding_dim),
            )
            self.distance_encoder = nn.Sequential(
                nn.Linear(4, geo_embedding_dim // 4),
                nn.GELU(),
                nn.Linear(geo_embedding_dim // 4, geo_embedding_dim // 2),
            )
        else:
            self.geo_in_dim = 0
            self.geo_encoder = None
            self.distance_encoder = None

        # Station gating (status soft/hard mask) and attention (per-station weighting)
        self.gate_mlp = nn.Sequential(
            nn.Linear(2, attention_hidden_units),
            nn.GELU(),
            nn.Linear(attention_hidden_units, 1),
        )
        self.attn_mlp = nn.Sequential(
            nn.Linear(2, attention_hidden_units),
            nn.GELU(),
            nn.Linear(attention_hidden_units, 1),
        ) if enable_station_attention else None

        self.input_dropout = nn.Dropout(input_dropout_rate)

        # Spatial compression: flatten stations then project to d_model
        spatial_in_dim = (
            num_stations * features_per_station
            + (num_stations * geo_embedding_dim if self.use_geo else 0)
        )
        self.spatial = nn.Sequential(
            nn.Linear(spatial_in_dim, model_dim),
            nn.GELU(),
        )
        self.spatial_ln = nn.LayerNorm(model_dim)

        # Positional encoding + Transformer encoder (Pre-LN)
        self.pos_enc = SinusoidalPositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout_rate,
            batch_first=True,
            norm_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Post projection and head
        self.post = nn.Sequential(
            nn.Linear(model_dim, post_hidden_units),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
        self.head = nn.Linear(post_hidden_units, 2 if enable_heteroscedastic_output else 1)

    def _compute_gates_and_weights(self, pgd_values: torch.Tensor, station_status: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute station gates (soft/hard) and optional attention weights.
        Returns (gate, weight) both in shape [B, T, N].
        """
        if self.hard_mask:
            gate = station_status
        else:
            gate_input = torch.stack([pgd_values, station_status], dim=-1)
            gate = torch.sigmoid(self.gate_mlp(gate_input)).squeeze(-1) * station_status
        if self.use_station_attention:
            attention_input = torch.stack([pgd_values, station_status], dim=-1)
            attention_logits = self.attn_mlp(attention_input).squeeze(-1)
            if self.normalize_station_attention:
                availability_mask = (station_status > 0.0)
                scaled_logits = attention_logits / max(self.attn_temperature, 1e-6)
                scaled_logits = torch.where(availability_mask, scaled_logits, torch.full_like(scaled_logits, float('-inf')))
                weight = torch.softmax(scaled_logits, dim=-1)
                weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)
            else:
                weight = torch.sigmoid(attention_logits)
        else:
            weight = torch.ones_like(gate)
        return gate, weight

    def forward(self, inputs: torch.Tensor, station_coords_deg: Optional[torch.Tensor] = None):
        batch_size, time_steps, feature_size = inputs.shape
        expected_feature_size = self.num_stations * self.feature_dim
        assert feature_size == expected_feature_size, f"Expected last dim {expected_feature_size}, got {feature_size}"

        inputs = inputs.view(batch_size, time_steps, self.num_stations, self.feature_dim)
        pgd_values = inputs[..., 0]
        station_status = inputs[..., 1]

        gate, weight = self._compute_gates_and_weights(pgd_values, station_status)
        pgd_values_mod = pgd_values * gate * weight
        if self.pgd_clip is not None:
            pgd_values_mod = torch.clamp(pgd_values_mod, -float(self.pgd_clip), float(self.pgd_clip))
        status_mod = station_status

        modified_inputs = torch.stack([pgd_values_mod, status_mod], dim=-1)

        if self.use_geo:
            if station_coords_deg is not None:
                if station_coords_deg.ndim != 3 or station_coords_deg.shape[0] != batch_size or station_coords_deg.shape[1] != self.num_stations or station_coords_deg.shape[2] != 2:
                    raise ValueError(f"station_coords_deg must be [B, N, 2] with N==num_stations; got {tuple(station_coords_deg.shape)}")

                flattened_coords_deg = station_coords_deg.reshape(batch_size * self.num_stations, 2)
                geo_feature_inputs = self._coords_to_features_with_mode(flattened_coords_deg, self.geo_mode)
                geo_embedding_flat = self.geo_encoder(geo_feature_inputs)
                geo_embedding_bn = geo_embedding_flat.view(batch_size, self.num_stations, self.geo_embed_dim)

                station_distance_stats = self._compute_station_distance_stats(station_coords_deg)
                distance_feature_embedding = self.distance_encoder(station_distance_stats)

                combined_geo_features = torch.cat([geo_embedding_bn, distance_feature_embedding], dim=-1)
                if combined_geo_features.shape[-1] != self.geo_embed_dim:
                    combined_geo_features = combined_geo_features[..., :self.geo_embed_dim]

                expanded_geo_features = combined_geo_features.unsqueeze(1).expand(batch_size, time_steps, self.num_stations, self.geo_embed_dim)
                modified_inputs = torch.cat([modified_inputs, expanded_geo_features], dim=-1)
            elif self.station_geo_embed is not None:
                expanded_geo_features = self.station_geo_embed.unsqueeze(0).unsqueeze(0).expand(batch_size, time_steps, self.num_stations, self.geo_embed_dim)
                modified_inputs = torch.cat([modified_inputs, expanded_geo_features], dim=-1)
            else:
                raise RuntimeError("Geo features enabled but station coordinates not provided. Pass station_coords_deg to forward() or call set_station_coords().")

        flattened_inputs = modified_inputs.view(batch_size, time_steps, -1)
        flattened_inputs = self.input_dropout(flattened_inputs)

        hidden_states = self.spatial(flattened_inputs)
        hidden_states = self.spatial_ln(hidden_states)
        hidden_states = self.pos_enc(hidden_states)
        if self.use_causal_mask:
            current_time_steps = hidden_states.size(1)
            mask = torch.full((current_time_steps, current_time_steps), float('-inf'), device=hidden_states.device)
            mask = torch.triu(mask, diagonal=1)
            hidden_states = self.encoder(hidden_states, mask=mask)
        else:
            hidden_states = self.encoder(hidden_states)

        post_activations = self.post(hidden_states)
        output_tensor = self.head(post_activations)

        if self.heteroscedastic:
            mean_pred = output_tensor[..., 0]
            log_variance = output_tensor[..., 1]
            return mean_pred, log_variance
        else:
            predictions = output_tensor.squeeze(-1)
            return predictions

    def set_station_coords(self, coords_deg: torch.Tensor):
        """Set station coordinates and build geo embedding.
        coords_deg: [N, 2] with (lat_deg, lon_deg). Expects N == num_stations.
        Registers buffer 'station_geo_embed' with shape [N, geo_embed_dim].
        """
        if not self.use_geo:
            return  # no-op when geo disabled
        if coords_deg.ndim != 2 or coords_deg.shape[0] != self.num_stations or coords_deg.shape[1] != 2:
            raise ValueError(f"coords_deg must be [num_stations, 2], got {tuple(coords_deg.shape)}")
        device = next(self.parameters()).device
        coords_deg = coords_deg.to(device=device, dtype=torch.float32)
        geo_feature_inputs = self._coords_to_features(coords_deg)
        embed = self.geo_encoder(geo_feature_inputs)
        # Store as buffer to move with model.to(device)
        try:
            self.register_buffer('station_geo_embed', embed)
        except Exception:
            # If already registered, just assign
            self.station_geo_embed = embed

    def _coords_to_features(self, coords_deg: torch.Tensor) -> torch.Tensor:
        """Convert [N,2] lat/lon degrees to feature set per station based on geo_mode."""
        return self._coords_to_features_with_mode(coords_deg, self.geo_mode)

    def _coords_to_features_with_mode(self, coords_deg: torch.Tensor, mode: str) -> torch.Tensor:
        """Convert lat/lon degrees to feature set, for either station or event mode."""
        lat_rad = coords_deg[:, 0] * (math.pi / 180.0)
        lon_rad = coords_deg[:, 1] * (math.pi / 180.0)
        if mode == 'xyz':
            x = torch.cos(lat_rad) * torch.cos(lon_rad)
            y = torch.cos(lat_rad) * torch.sin(lon_rad)
            z = torch.sin(lat_rad)
            return torch.stack([x, y, z], dim=-1)
        elif mode == 'sin_cos':
            return torch.stack([torch.sin(lat_rad), torch.cos(lat_rad), torch.sin(lon_rad), torch.cos(lon_rad)], dim=-1)
        else:  # 'raw'
            lat_norm = coords_deg[:, 0] / 90.0
            lon_norm = coords_deg[:, 1] / 180.0
            return torch.stack([lat_norm, lon_norm], dim=-1)

    def _compute_station_distance_stats(self, station_coords_deg: torch.Tensor) -> torch.Tensor:
        """
        计算台站间距离统计特征，用于捕获台站网络的几何配置信息
        Args:
            station_coords_deg: [B, N, 2] 台站坐标 (lat_deg, lon_deg)
        Returns:
            distance_stats: [B, N, 4] 每个台站的距离统计特征
        """
        batch_size, num_stations, _ = station_coords_deg.shape

        lat_rad = station_coords_deg[..., 0] * (math.pi / 180.0)
        lon_rad = station_coords_deg[..., 1] * (math.pi / 180.0)

        lat1 = lat_rad.unsqueeze(2)
        lon1 = lon_rad.unsqueeze(2)
        lat2 = lat_rad.unsqueeze(1)
        lon2 = lon_rad.unsqueeze(1)

        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1
        haversine_a = torch.sin(delta_lat/2)**2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(delta_lon/2)**2
        central_angle = 2 * torch.asin(torch.sqrt(torch.clamp(haversine_a, 0, 1)))
        pairwise_distances_km = 6371.0 * central_angle

        identity = torch.eye(num_stations, device=pairwise_distances_km.device, dtype=pairwise_distances_km.dtype)
        pairwise_distances_km = pairwise_distances_km + identity.unsqueeze(0) * 1e6

        min_distance = torch.min(pairwise_distances_km, dim=-1)[0]
        max_distance = torch.max(pairwise_distances_km, dim=-1)[0]
        mean_distance = torch.mean(pairwise_distances_km, dim=-1)
        std_distance = torch.std(pairwise_distances_km, dim=-1)

        distance_stats = torch.stack([min_distance, max_distance, mean_distance, std_distance], dim=-1)
        distance_stats = distance_stats / 1000.0

        return distance_stats


def build_ct_model(
    num_stations: int,
    features_per_station: int = 2,
    model_dim: int = 128,
    num_attention_heads: int = 4,
    num_encoder_layers: int = 2,
    feedforward_dim: int = 256,
    dropout_rate: float = 0.2,
    post_hidden_units: int = 128,
    input_dropout_rate: float = 0.0,
    enable_station_attention: bool = True,
    attention_hidden_units: int = 16,
    normalize_station_attention: bool = False,
    attention_temperature: float = 1.0,
    enable_hard_mask: bool = False,
    enable_heteroscedastic_output: bool = True,
    enable_causal_mask: bool = False,
    enable_geo_features: bool = False,
    geo_feature_mode: str = "xyz",
    geo_embedding_dim: int = 16,
    pgd_value_clip: float | None = None,
    **legacy_kwargs: Any,
) -> CTSequenceModel:
    alias = {
        "feature_dim": "features_per_station",
        "d_model": "model_dim",
        "nhead": "num_attention_heads",
        "num_layers": "num_encoder_layers",
        "ffn_dim": "feedforward_dim",
        "dropout": "dropout_rate",
        "post_units": "post_hidden_units",
        "input_dropout": "input_dropout_rate",
        "use_station_attention": "enable_station_attention",
        "attn_hidden": "attention_hidden_units",
        "attn_temperature": "attention_temperature",
        "hard_mask": "enable_hard_mask",
        "heteroscedastic": "enable_heteroscedastic_output",
        "use_causal_mask": "enable_causal_mask",
        "use_geo": "enable_geo_features",
        "geo_mode": "geo_feature_mode",
        "geo_embed_dim": "geo_embedding_dim",
        "pgd_clip": "pgd_value_clip",
    }
    params = {
        "num_stations": num_stations,
        "features_per_station": features_per_station,
        "model_dim": model_dim,
        "num_attention_heads": num_attention_heads,
        "num_encoder_layers": num_encoder_layers,
        "feedforward_dim": feedforward_dim,
        "dropout_rate": dropout_rate,
        "post_hidden_units": post_hidden_units,
        "input_dropout_rate": input_dropout_rate,
        "enable_station_attention": enable_station_attention,
        "attention_hidden_units": attention_hidden_units,
        "normalize_station_attention": normalize_station_attention,
        "attention_temperature": attention_temperature,
        "enable_hard_mask": enable_hard_mask,
        "enable_heteroscedastic_output": enable_heteroscedastic_output,
        "enable_causal_mask": enable_causal_mask,
        "enable_geo_features": enable_geo_features,
        "geo_feature_mode": geo_feature_mode,
        "geo_embedding_dim": geo_embedding_dim,
        "pgd_value_clip": pgd_value_clip,
    }
    for old_name, new_name in alias.items():
        if old_name in legacy_kwargs:
            params[new_name] = legacy_kwargs.pop(old_name)
    return CTSequenceModel(**params)
