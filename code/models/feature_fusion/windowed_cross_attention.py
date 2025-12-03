import torch
import torch.nn as nn

from models.feature_fusion.window_attention import WindowAttention
from models.feature_fusion.multi_window_attention import MultiWindowAttention
from models.feature_fusion.cross_attention import CrossAttention


class WindowedCrossAttention(nn.Module):
    """
    WCA: Windowed Cross-Attention = WA + MWA + CA
    """

    def __init__(
        self,
        img_dim,
        iqap_dim,
        embed_dim,
        num_heads,
        window_sizes,
        swin_params,
        seq_length,
        module_flags=None,
        block_layers=1,
    ):
        super(WindowedCrossAttention, self).__init__()

        self.flags = module_flags or {}
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        self.window_sizes = window_sizes
        self.block_layers = block_layers

        self.input_embed_img = nn.Linear(img_dim, embed_dim)
        self.input_embed_iqap = nn.Linear(iqap_dim, embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        if self._use_module("use_wa", default=True):
            self.window_attention = WindowAttention(
                embed_dim=embed_dim,
                num_layers=swin_params["num_layers"],
                window_sizes=swin_params["window_sizes"],
                num_heads=swin_params["num_heads"],
                mlp_ratio=swin_params.get("mlp_ratio", 4.0),
                drop=swin_params.get("drop", 0.0),
            )

        if self._use_module("use_mwa", default=True):
            self.multi_window_attention = MultiWindowAttention(
                input_dim=embed_dim,
                output_dim=embed_dim,
                seq_length=seq_length,
                window_sizes=window_sizes,
                num_heads=num_heads,
            )

        if self._use_module("use_ca", default=True):
            self.cross_attentions = nn.ModuleList(
                [CrossAttention(embed_dim=embed_dim, num_heads=num_heads) for _ in range(swin_params["num_layers"])]
            )

        self.enabled_modules = []
        if self._use_module("use_wa", default=True):
            self.enabled_modules.append("WA")
        if self._use_module("use_mwa", default=True):
            self.enabled_modules.append("MWA")
        if self._use_module("use_ca", default=True):
            self.enabled_modules.append("CA")
        self.num_modules = len(self.enabled_modules)

        self.theta = nn.Parameter(torch.zeros(self.num_modules))

    def _use_module(self, flag_name, default):
        legacy_map = {
            "use_wa": "use_branch1",
            "use_mwa": "use_branch2",
            "use_ca": "use_branch3",
        }
        if flag_name in self.flags:
            return self.flags[flag_name]
        legacy_key = legacy_map.get(flag_name)
        if legacy_key is not None and legacy_key in self.flags:
            return self.flags[legacy_key]
        return default

    def mean_pooling(self, x, target_length):
        seq_len = x.shape[1]
        if seq_len == target_length:
            return x

        step = seq_len // target_length
        pooled = []
        for i in range(target_length):
            start = i * step
            end = (i + 1) * step if i != target_length - 1 else seq_len
            pooled.append(x[:, start:end, :].mean(dim=1))
        return torch.stack(pooled, dim=1)

    def forward(self, img, iqap):
        x_img = self.input_embed_img(img)
        x_iqap = self.input_embed_iqap(iqap)
        x = torch.cat([x_img, x_iqap], dim=1)
        x = self.mean_pooling(x, self.seq_length)

        for _ in range(self.block_layers):
            wa_out = None
            if self._use_module("use_wa", default=True):
                wa_outputs = self.window_attention(x)
                stacked1 = torch.stack(wa_outputs, dim=0)
                wa_out = stacked1.mean(dim=0)

            mwa_out = None
            if self._use_module("use_mwa", default=True):
                mwa_out = self.multi_window_attention(x)

            ca_out = None
            if self._use_module("use_ca", default=True):
                feats3 = []
                for idx, cross_attn in enumerate(self.cross_attentions):
                    wa_feat = wa_outputs[idx] if idx < len(wa_outputs) else torch.zeros_like(x)
                    feats3.append(cross_attn(wa_feat))
                stacked3 = torch.stack(feats3, dim=0)
                ca_out = stacked3.mean(dim=0)

            final_features = [wa_out, mwa_out, ca_out]
            feats = [f for f in final_features if f is not None]

            stacked = torch.stack(feats, dim=0)
            weights = torch.softmax(self.theta, dim=0)
            weights = weights.view(self.num_modules, 1, 1, 1)
            x = (stacked * weights).sum(dim=0)

        out = x.mean(dim=1)
        out = self.mlp(out)
        return out

