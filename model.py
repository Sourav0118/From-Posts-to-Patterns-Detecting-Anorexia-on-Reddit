import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PatchTSTConfig, PatchTSTForClassification

class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.attention = nn.MultiheadAttention(768, 12)
        self.fc1 = nn.Linear(in_features=768, out_features=384)
        self.fc2 = nn.Linear(in_features=384, out_features=128)
        self.fc3 = nn.Linear(in_features=128, out_features=32)
        self.fc4 = nn.Linear(in_features=32, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(0)
        x, _ = self.attention(x, x, x)
        x = x.squeeze(0)
        x = torch.relu(self.fc2(torch.relu(self.fc1(x))))
        # Perform global pooling over the batch dimension
        x = torch.relu(self.fc3(x))
        x = torch.mean(x, dim=0, keepdim=True)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x
    

# classification task with two input channel2 and 3 classes
config = PatchTSTConfig(
    num_input_channels=2,
    num_targets=1,
    context_length=2000,
    patch_length=12,
    stride=12,
    use_cls_token=True,
    # loss='bce',
    head_dropout=0.5,
)
