import torch 
from torch.nn.functional import softmax

def manual_softmax(x):
    """
    x:Tensor[B,S,D]
    """
    m = torch.max(x, dim=-1, keepdim=True,)[0] #返回最大值
    exp_x = torch.exp(x - m)  # 减去最大值避免数值溢出
    denominator = torch.sum(exp_x, dim=-1, keepdim=True)
    return exp_x / denominator


if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(5,8,10)
    manual_result = manual_softmax(x)
    origin_result = softmax(x, dim=-1)
    
    is_close = torch.allclose(manual_result, origin_result, rtol=1e-5, atol=1e-8)
    print(f"Result Dim {origin_result.size()}")
    print(f"比较结果: {is_close}")