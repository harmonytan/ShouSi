import torch
import torch.nn as nn
import torch.nn.functional as F

class GQA(nn.Module):
    def __init__(self,d_model,head_num,group_num):
        super(GQA,self).__init__()
        self.d_model = d_model
        self.head_num = head_num #多头头数
        self.group_num = group_num #每组头数
        self.group = self.head_num//self.group_num #组数

        assert self.d_model%self.head_num == 0 ,'d_model must be divisible by head_num'
        assert self.head_num%self.group_num == 0 ,'head_num must be divisible by group_num'
        self.d_head = self.d_model//self.head_num
        self.d_group = self.d_head * self.group

        self.q_proj = nn.Linear(self.d_model,self.d_model)
        self.k_proj = nn.Linear(self.d_model,self.d_group)
        self.v_proj = nn.Linear(self.d_model,self.d_group)
        self.o_proj = nn.Linear(self.d_model,self.d_model)
        
    def forward(self,x,test):
        batch_size, seq_len, _ = x.size()
        q = self.q_proj(x).view(batch_size,seq_len,self.group_num,self.group,self.d_head).permute(0,2,3,1,4)

        k = self.k_proj(x).view(batch_size,seq_len,self.group,self.d_head).unsqueeze(-2).expand(-1,-1,self.group_num,-1,-1).permute(0,2,3,1,4)
        v = self.v_proj(x).view(batch_size,seq_len,self.group,self.d_head).unsqueeze(-2).expand(-1,-1,self.group_num,-1,-1).permute(0,2,3,1,4)
    
        attn = torch.matmul(q,k.transpose(-1,-2))/(self.d_head**0.5)
        attn = F.softmax(attn,dim=-1)
        attn = torch.matmul(attn,v)
        attn = attn.permute(0,3,1,2,4).contiguous().view(batch_size,seq_len,self.d_model)
        return self.o_proj(attn)





        