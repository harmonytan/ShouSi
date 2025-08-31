import torch 

def rotate_half(x):
    x1,x2 = torch.chunk(x,2,dim=-1)
    return torch.cat([-x2,x1],dim=-1)

def apply_rope_x(x,sin,cos):
    return x*cos + rotate_half(x)*sin

class MLA(torch.nn.Module):
    def __init__(self, d_model, n_heads,max_len=1024,theta=10000.0):
        super().__init__()

        self.n_heads = n_heads
        assert d_model//n_heads

        self.d_model = d_model
        self.d_h = d_model//n_heads

        self.qproj_dim = d_model // 2
        self.kvproj_dim = d_model // 2
        self.qk_rope_dim = self.d_h // 2
        self.qk_nope_dim = self.d_h // 2
        
        #Qproj
        self.W_DQ = torch.nn.Parameter(0.01*torch.randn(self.d_model, self.qproj_dim))
        self.W_UQ = torch.nn.Parameter(0.01*torch.randn(self.qproj_dim,self.d_model))

        #KVproj 
        self.W_DKV = torch.nn.Parameter(0.01*torch.randn(self.d_model,self.kvproj_dim))
        self.W_UK = torch.nn.Parameter(0.01*torch.randn(self.kvproj_dim,self.n_heads*self.qk_nope_dim))
        self.W_UV = torch.nn.Parameter(0.01*torch.randn(self.kvproj_dim,self.d_model))
        self.W_K_rope = torch.nn.Parameter(0.01*torch.randn(self.d_model,self.qk_rope_dim))

        #rope 
        self.max_len = max_len
        self.theta = theta
        i = torch.arange(0, self.d_h // 2, dtype=torch.float32)  # 生成索引 [0, 1, ..., d_h//2 -1]
        theta_i = self.theta ** (2 * i / self.d_h)  # 计算 θ^(2i/d_h)
        freqs = 1.0 / theta_i  # 取倒数
        emb = torch.outer(torch.arange(self.max_len).float(), freqs)
        sin_cached = emb.sin()[None,None,:,:]
        cos_cached = emb.cos()[None,None,:,:]

        self.register_buffer('sin_cached',sin_cached)
        self.register_buffer('cos_cached',cos_cached)

        #Oproj
        self.o_proj = torch.nn.Linear(self.d_model,self.d_model)

    def forward(self,x,past_length=0):
        B,S,D = x.size()

        #cal Q
        c_q = x @ self.W_DQ
        q = c_q @ self.W_UQ
        q = q.view(B,S,self.n_heads,self.d_h).transpose(-2,-3) #(B,H,S,d_h)
        q_nope, q_rope = torch.split(q,[self.qk_nope_dim, self.qk_rope_dim],dim=-1) #(B,H,S,d_rope)

        #cal KV 
        c_kv = x @ self.W_DKV
        k_nope = c_kv @ self.W_UK #(B,S,d_nope*H)
        k_rope = x @ self.W_K_rope #(B,S,d_rope)
        v = c_kv @ self.W_UV

        k_nope = k_nope.view(B,S,self.n_heads, self.qk_nope_dim).transpose(-2,-3) #(B,H,S,d_nope)
        v_head = v.view(B,S,self.n_heads,self.d_h).transpose(-2,-3) #(B,H,S,d_h)

        #apply rope Q 
        cos_q = self.cos_cached[:, :, past_length:past_length+S, :self.qk_rope_dim//2].repeat(1, 1, 1, 2)
        sin_q = self.sin_cached[:, :, past_length:past_length+S, :self.qk_rope_dim//2].repeat(1, 1, 1, 2)

        q_rope = apply_rope_x(q_rope,sin_q,cos_q)
        q_head = torch.cat([q_nope,q_rope],dim=-1) #(B,H,S,d_h)

        #apply rope K 
        cos_k = self.cos_cached[:,:,past_length:past_length+S,:self.qk_rope_dim//2].repeat(1,1,1,2)
        sin_k = self.sin_cached[:, :, past_length:past_length+S, :self.qk_rope_dim//2].repeat(1, 1, 1, 2)

        k_rope = apply_rope_x(k_rope,sin_k,cos_k).expand(-1,self.n_heads,-1,-1)
        k_head = torch.cat([k_nope,k_rope],dim=-1)  #(B,H,S,d_h)

        #cal attention 
        attention = torch.nn.functional.scaled_dot_product_attention(q_head,k_head,v_head)

        attention = attention.transpose(-2,-3).contiguous().view(B,S,D)

        output = self.o_proj(attention)
        return output
    
if __name__ == "__main__":
    model = MLA(20,5)
    B,S,D = 1,30,20
    x = torch.randn([B ,S ,D] )
    print(model(x))



 


        

