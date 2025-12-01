import torch
import torch.nn as nn


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x):
        # x is of shape (B, C, H, W)
        x = self.dwconv(x)
        return x


class CGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)

        # Replace nn.Linear with 1x1 Conv2d for 4D input
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout2d(drop)

    def forward(self, x):
        residual = x  # Save the input for residual connection

        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = torch.sigmoid(x)  # Apply sigmoid to attention weights

        # Multiply attention weights with original input
        x = x * residual
        return x


# Testing the modified CGLU with residual connection and 4D input
if __name__ == "__main__":
    B, C, H, W = 1, 768, 16, 16
    x = torch.randn(B, C, H, W)
    model = CGLU(in_features=C)
    output = model(x)
    print("Output shape:", output.shape)
