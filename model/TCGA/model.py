from torch import nn



class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        self.label_mlp =  nn.Linear(time_emb_dim, out_ch)
        self.label_emb = nn.Embedding(2, time_emb_dim)
        if up:
            self.conv1 = nn.Conv1d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose1d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv1d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.conv3 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.conv4 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm1d(out_ch)
        self.bnorm2 = nn.BatchNorm1d(out_ch)
        self.bnorm3 = nn.BatchNorm1d(out_ch)
        self.bnorm4 = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU()
        self.mha = nn.MultiheadAttention(1024, 8,batch_first=True)
        
    def forward(self, x, t,y):
        if x.shape[2]<16:
            time_emb = self.relu(self.time_mlp(t))
            time_emb = time_emb[(..., ) + (None, ) * 1]
            cls_emb = self.label_emb(y)
            cls_emb = self.relu(self.label_mlp(cls_emb))
            cls_emb = cls_emb[(..., ) + (None, ) * 1]
            
            h_0 = self.bnorm1(self.relu(self.conv1(x)))
            h = h_0*cls_emb + time_emb
            h = self.bnorm2(self.relu(self.conv2(h)))

            
            h_att = torch.permute(h, (0, 2, 1))
            h_att, _ = self.mha(h_att,h_att,h_att)
            h_att = torch.permute(h_att, (0, 2, 1))
            h = h_att+h
        # Down or Upsample
            return self.transform(h)
        else:
            time_emb = self.relu(self.time_mlp(t))
            time_emb = time_emb[(..., ) + (None, ) * 1]
            cls_emb = self.label_emb(y)
            cls_emb = self.relu(self.label_mlp(cls_emb))
            cls_emb = cls_emb[(..., ) + (None, ) * 1]
            
            h_0 = self.bnorm1(self.relu(self.conv1(x)))
            h = h_0*cls_emb + time_emb
            h = self.bnorm2(self.relu(self.conv2(h)))



        # Down or Upsample
            return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = 2
        down_channels = (64,128,256,512,512,1024,1024) #(64, 128, 256, 512, 1024)
        up_channels = (1024,1024,512,512,256,128,64)#(1024, 512, 256, 128, 64)
        out_dim = 1
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
#         self.embedding = 
        
        # Initial projection
        self.conv0 = nn.Conv1d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])

        self.output = nn.Conv1d(up_channels[-1], 2,1)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x, timestep,y):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t,y)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t,y)
        return self.output(x)
