import torch
import torch.nn as nn


def get_model(name: str, device: str, in_feats: int, out_feats: int = 2) -> nn.Module:
    if name == "mlp2":
        return nn.Sequential(
            nn.Linear(in_feats, 50),
            nn.ReLU(),
            nn.Linear(50, out_feats),
        ).to(device)
    elif name == "mlp4":  # TODO Actually is MLP3
        return nn.Sequential(
            nn.Linear(in_feats, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, out_feats),
        ).to(device)
    elif name == "mlp5":
        return nn.Sequential(
            nn.Linear(in_feats, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 100),
            nn.ReLU(),
            nn.Linear(100, out_feats),
        ).to(device)
    elif name == "mlp_4_512":
        return nn.Sequential(
            nn.Linear(in_feats, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, out_feats),
        ).to(device)
    elif name == "double_input_mlp":
        return DoubleInputMLP(in_feats, out_feats).to(device)
    else:
        raise RuntimeError(f"Unknown model: {name}")


class DoubleInputMLP(nn.Module):
    def __init__(self, in_feats: int, out_feats: int = 2):
        super().__init__()
        print("Using DoubleInputMLP")
        self.mlp4 = nn.Sequential(
            nn.Linear(2 * in_feats, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, out_feats),
        )
        self.in_feats = in_feats
        self.out_feats = out_feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = torch.cat((x, x), dim=1)
        return self.mlp4(x2)

    def get_in_feats(self) -> int:
        return self.in_feats

    def get_out_feats(self) -> int:
        return self.out_feats
