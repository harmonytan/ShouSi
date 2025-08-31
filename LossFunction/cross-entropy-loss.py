import torch 
import torch.nn as nn 
from torch.nn import CrossEntropyLoss

class SimpleNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(10,20)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(20,10)
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

def cross_entropy_loss(input, labels):
    """
    input: Tensor[B,N,V] - 原始 logits
    labels: Tensor[B,N]
    """
    # 使用 log_softmax 避免数值不稳定
    log_probs = torch.log_softmax(input, dim=-1)
    selected_log_probs = torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    loss = -torch.mean(selected_log_probs)
    return loss 

if __name__ == "__main__":
    model  = SimpleNetwork()
    torch.manual_seed(42)
    x = torch.randn(5,8,10)
    output = model(x)
    labels = torch.randint(0,10,(5,8))
    
    print("手动实现的交叉熵损失:")
    print(cross_entropy_loss(output,labels))
    
    print("\n使用PyTorch内置的CrossEntropyLoss:")
    criterion = CrossEntropyLoss()
    print(criterion(output.view(-1, output.size(-1)), labels.view(-1)))
    
