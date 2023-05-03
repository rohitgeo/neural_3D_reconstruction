import numpy as np
import torch
import torch.nn as nn
from config import use_custom_components

# sin-cose embedding module
class Embedder(nn.Module):
    def __init__(self, **kwargs):
        super(Embedder, self).__init__()
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs["input_dims"]
        out_dim = 0
        if self.kwargs["include_input"]:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs["max_freq_log2"]
        N_freqs = self.kwargs["num_freqs"]

        if self.kwargs["log_sampling"]:
            freq_bands = 2.0 ** torch.linspace(0.0, max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2.0**0.0, 2.0**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs["periodic_fns"]:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def forward(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class FeedForward(nn.Module):
    def __init__(self, dim, hid_dim, dp_rate):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, dim)
        self.dp = nn.Dropout(dp_rate)
        self.activ = nn.ReLU()

    def forward(self, x):
        x = self.dp(self.activ(self.fc1(x)))
        x = self.dp(self.fc2(x))
        return x


# Subtraction-based efficient attention
class Attention2D(nn.Module):
    def __init__(self, dim, dp_rate):
        super(Attention2D, self).__init__()
        self.q_fc = nn.Linear(dim, dim, bias=False)
        self.k_fc = nn.Linear(dim, dim, bias=False)
        self.v_fc = nn.Linear(dim, dim, bias=False)
        self.pos_fc = nn.Sequential(
            nn.Linear(4, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, dim),
        )
        self.attn_fc = nn.Sequential(
            nn.Linear(dim, dim // 8),
            nn.ReLU(),
            nn.Linear(dim // 8, dim),
        )
        self.out_fc = nn.Linear(dim, dim)
        self.dp = nn.Dropout(dp_rate)

    def forward(self, q, k, pos, mask=None):
        q = self.q_fc(q)
        k = self.k_fc(k)
        v = self.v_fc(k)

        pos = self.pos_fc(pos)
        attn = k - q[:, :, None, :] + pos
        attn = self.attn_fc(attn)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(attn, dim=-2)
        attn = self.dp(attn)

        x = ((v + pos) * attn).sum(dim=2)
        x = self.dp(self.out_fc(x))
        return x


class CustomTransformer2D(nn.Module):
    def __init__(self, atten_d, ffn_d):
        super(CustomTransformer2D, self).__init__()
        self.fq = nn.Linear(atten_d, atten_d)
        self.fk = nn.Linear(atten_d, atten_d)
        self.fv = nn.Linear(atten_d, atten_d)
        self.fp = nn.Linear(4, atten_d)
        self.fa = nn.Linear(atten_d, atten_d)
        self.fo = nn.Linear(atten_d, atten_d)

        self.norm = nn.LayerNorm(atten_d)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-2)

        self.ffn_norm = nn.LayerNorm(atten_d)
        self.ffn1 = nn.Linear(atten_d, ffn_d)
        self.ffn2 = nn.Linear(ffn_d, atten_d)

    def forward(self, q, k, pos, mask=None):
        # Cross Attention
        q_res = q
        q = self.norm(q)
        q = self.fq(q)

        k = self.fk(k)
        v = self.fv(k)
        pos = self.fp(pos)

        a = k - q[:, :, None, :] + pos
        a = self.softmax(a)

        o = torch.diagonal(torch.matmul(a.permute(0, 1, 3, 2), v), dim1=-2, dim2=-1)
        o = self.fo(o)

        o = q_res + o

        # FFN
        o_res = o

        o = self.ffn_norm(o)
        o = self.ffn1(o)
        o = self.relu(o)
        o = self.ffn2(o)

        o = o + o_res
        return o


# View Transformer
class Transformer2D(nn.Module):
    def __init__(self, dim, ff_hid_dim, ff_dp_rate, attn_dp_rate):
        super(Transformer2D, self).__init__()
        self.attn_norm = nn.LayerNorm(dim, eps=1e-6)
        self.ff_norm = nn.LayerNorm(dim, eps=1e-6)

        self.ff = FeedForward(dim, ff_hid_dim, ff_dp_rate)
        self.attn = Attention2D(dim, attn_dp_rate)

    def forward(self, q, k, pos, mask=None):
        residue = q
        x = self.attn_norm(q)
        x = self.attn(x, k, pos, mask)
        x = x + residue

        residue = x
        x = self.ff_norm(x)
        x = self.ff(x)
        x = x + residue

        return x






# attention module for self attention.
# contains several adaptations to incorportate positional information (NOT IN PAPER)
#   - qk (default) -> only (q.k) attention.
#   - pos -> replace (q.k) attention with position attention.
#   - gate -> weighted addition of  (q.k) attention and position attention.
class Attention(nn.Module):
    def __init__(self, dim, n_heads, dp_rate, attn_mode="qk", pos_dim=None):
        super(Attention, self).__init__()

        #creating a self var so it can be accessed from forward
        self.input_dim = dim
        self.input_heads = n_heads

        #Layers for q,v, and k
        # why am I diving by the number of heads?
        self.q_linear = nn.Linear(dim, dim // n_heads)
        self.k_linear = nn.Linear(dim, dim // n_heads)
        self.v_linear = nn.Linear(dim, dim)


        #drop out layer for regularization
        self.layer_drop = nn.Dropout(dp_rate)

        #create the softmax layer
        self.layer_softmax = nn.Softmax()

        #Handling the 3 methods: Positional info

        #QK mode

        #pos mode

        #gate mode
    

    def forward(self, x, pos=None, ret_attn=False):


      #apply q,k and v
      q = self.q_linear(x)
      k = self.k_linear(x)
      v = self.v_linear(x)

      #q shape
      #[2275, 64, 16]

      #re-shpaing to handle multiple heads
      #Q should become, (batch_size, seq_len, n_heads, head_dim)
      #print("q shape",q.shape)

      #Q
      q_reshaped = q.view(q.size(0),q.size(1),self.input_heads,-1)
      q_transp = q_reshaped.transpose(1,2)

      #K
      k_reshaped = k.view(k.size(0),k.size(1),self.input_heads,-1)
      k_transp = k_reshaped.transpose(1,2)


      #V
      v_reshaped = v.view(v.size(0),v.size(1),self.input_heads,-1)
      v_transp = v_reshaped.transpose(1,2)

      #prepare data for the matrix multiplication
      dim_heads = self.input_dim/self.input_heads

      #divide q by its square root?
      q_prep = q_transp/torch.sqrt(torch.tensor(dim_heads).float())

      #performing another transpose on K, might be the wrong item in the spot, since you have multiple values of 4
      k_transp2 = k_transp.transpose(2,3)


      ############ATTENTION####################
      #compute the dot product between q and k
      attn = torch.matmul(q_prep,k_transp2)

      #now apply a dropout layer
      attn_drop = self.layer_drop(attn)

      #perform a softmax function call
      attn_soft = self.layer_softmax(attn_drop)

      ## Print out the attention from softmax
      

      #multipily  attention scores by v to get outptut
      #v typical dimensions:

      output = torch.matmul(attn_soft,v_transp)
      output_concat = torch.cat([output[:, i, :, :] for i in range(self.input_heads)], dim=2)


      #output_concat = output_concat.transpose(1,2)

      if ret_attn == True:
        return output_concat,attn_soft
      
      else:
        return output_concat




# light ray Transformer
class Transformer(nn.Module):
    def __init__(self, dim, ff_hid_dim, ff_dp_rate, n_heads, attn_dp_rate, attn_mode="qk", pos_dim=None):

        #needed for defining the nn.module?
        super(Transformer, self).__init__()

        #create layer normalization
        self.layer_norm = nn.LayerNorm(normalized_shape=dim)

        #create the multi-headed attention
        self.layer_MHA  = Attention(dim ,n_heads,attn_dp_rate, attn_mode, pos_dim)

        #post multi head attention norm
        self.layer_norm_post_MHA = nn.LayerNorm(normalized_shape=dim)

        #calling the pre-defined feed Forward
        self.layer_FF = FeedForward( dim,ff_hid_dim, ff_dp_rate)


    def forward(self, x, pos=None, ret_attn=False):

            #call layer norm on x
            x_norm = self.layer_norm(x)
            #print("return attention boolean",ret_attn)

            
            if ret_attn == True:
              #pass the x through the multi-headed attention
              x_att, attn_weights = self.layer_MHA(x_norm,pos,ret_attn)

            else:
              x_att = self.layer_MHA(x_norm,pos,ret_attn)

            #GENERATE ATTENTION MAPS]

            #print("shape of returned attention map",x_att.shape)
            
            #TESTING POST ATTENTION
            #print("Attention output: x attention shape",x_att.shape)
            #print("attention weights",attn_weights)

            #after MHA perform layer nrom
            x_norm2 = self.layer_norm_post_MHA(x_att)

            #call the feed forward function
            x_FF = self.layer_FF(x_norm2)

            #Now that I've gone through the feed fowrad layer, I need to add this back to my x
            #TEST
            #print("x_FF shape",x_FF.shape)
            #rint("x shape",x.shape)

            #TEST
            x_residue_result = x_FF + x

            #preparing the return value
            x_ret = x_residue_result

            #Return
            if ret_attn == True:

              return(x_ret,attn_weights)
            
            else:
              return(x_ret)

            
##### My Code

class GNT(nn.Module):
    def __init__(self, args, in_feat_ch=32, posenc_dim=3, viewenc_dim=3, ret_alpha=False):
        super(GNT, self).__init__()
        self.rgbfeat_fc = nn.Sequential(
            nn.Linear(in_feat_ch + 3, args.netwidth),
            nn.ReLU(),
            nn.Linear(args.netwidth, args.netwidth),
        )

        # NOTE: Apologies for the confusing naming scheme, here view_crosstrans refers to the view transformer, while the view_selftrans refers to the ray transformer
        
        self.view_selftrans = nn.ModuleList([])
        self.view_crosstrans = nn.ModuleList([])
        self.q_fcs = nn.ModuleList([])

        
        for i in range(args.trans_depth):
            # view transformer
            if use_custom_components:
                view_trans = CustomTransformer2D(
                    atten_d=args.netwidth,
                    ffn_d=int(args.netwidth * 4)
                )
            else:
                view_trans = Transformer2D(
                    dim=args.netwidth,
                    ff_hid_dim=int(args.netwidth * 4),
                    ff_dp_rate=0.1,
                    attn_dp_rate=0.1,
                )
            self.view_crosstrans.append(view_trans)
            # ray transformer
            ray_trans = Transformer(
                dim=args.netwidth,
                ff_hid_dim=int(args.netwidth * 4),
                n_heads=4,
                ff_dp_rate=0.1,
                attn_dp_rate=0.1,
            )
            self.view_selftrans.append(ray_trans)
            # mlp
            if i % 2 == 0:
                q_fc = nn.Sequential(
                    nn.Linear(args.netwidth + posenc_dim + viewenc_dim, args.netwidth),
                    nn.ReLU(),
                    nn.Linear(args.netwidth, args.netwidth),
                )
            else:
                q_fc = nn.Identity()
            self.q_fcs.append(q_fc)

        self.posenc_dim = posenc_dim
        self.viewenc_dim = viewenc_dim
        self.ret_alpha = ret_alpha
        self.norm = nn.LayerNorm(args.netwidth)
        self.rgb_fc = nn.Linear(args.netwidth, 3)
        self.relu = nn.ReLU()
        self.pos_enc = Embedder(
            input_dims=3,
            include_input=True,
            max_freq_log2=9,
            num_freqs=10,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )
        self.view_enc = Embedder(
            input_dims=3,
            include_input=True,
            max_freq_log2=9,
            num_freqs=10,
            log_sampling=True,
            periodic_fns=[torch.sin, torch.cos],
        )

    def forward(self, rgb_feat, ray_diff, mask, pts, ray_d):
        # compute positional embeddings
        viewdirs = ray_d
        viewdirs = viewdirs / torch.norm(viewdirs, dim=-1, keepdim=True)
        viewdirs = torch.reshape(viewdirs, [-1, 3]).float()
        viewdirs = self.view_enc(viewdirs)
        pts_ = torch.reshape(pts, [-1, pts.shape[-1]]).float()
        pts_ = self.pos_enc(pts_)
        pts_ = torch.reshape(pts_, list(pts.shape[:-1]) + [pts_.shape[-1]])
        viewdirs_ = viewdirs[:, None].expand(pts_.shape)
        embed = torch.cat([pts_, viewdirs_], dim=-1)
        input_pts, input_views = torch.split(embed, [self.posenc_dim, self.viewenc_dim], dim=-1)

        #SETTING UP TO ALWAYS RETURN ATTENTION
        #self.ret_alpha = True
        

        # project rgb features to netwidth
        rgb_feat = self.rgbfeat_fc(rgb_feat)
        # q_init -> maxpool
        q = rgb_feat.max(dim=2)[0]

        # transformer modules
        for i, (crosstrans, q_fc, selftrans) in enumerate(
            zip(self.view_crosstrans, self.q_fcs, self.view_selftrans)
        ):
            # view transformer to update q
            q = crosstrans(q, rgb_feat, ray_diff, mask)
            # embed positional information
            if i % 2 == 0:
                q = torch.cat((q, input_pts, input_views), dim=-1)
                q = q_fc(q)
                
            # ray transformer
            q = selftrans(q, ret_attn=self.ret_alpha)
            # 'learned' density
            if self.ret_alpha:
                q, attn = q
                print("Attention is",attn.shape)
        # normalize & rgb
        h = self.norm(q)
        outputs = self.rgb_fc(h.mean(dim=1))
        if self.ret_alpha:
            return torch.cat([outputs, attn], dim=1)
        else:
            return outputs
