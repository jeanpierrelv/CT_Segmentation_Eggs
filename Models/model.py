import torch
import torch.nn as nn
import numpy as np
from torchvision.transforms import CenterCrop
from torchvision.models import resnet18

print(torch.cuda.is_available())
print(torch.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# FULLY CONVOLUTIONAL NETWORKS (FCN)
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# FUN��ES AUXILIARES
#-------------------------------------------------------------------------------

def upsampling_weights(in_channels,out_channels,ker_size):
    """
        Cria um kernel 2D bilinear para upsampling.
    """
    f = (ker_size+1)//2
    if ker_size%2 == 1:
        center = f-1
    else:
        center = f-0.5
    og = np.ogrid[:ker_size,:ker_size]
    filtr = (1-abs(og[0]-center)/f)*(1-abs(og[1]-center)/f)
    w = np.zeros((in_channels,out_channels,ker_size,ker_size),dtype=np.float64)
    w[range(in_channels),range(out_channels),:,:] = filtr
    return torch.from_numpy(w).float()

#-------------------------------------------------------------------------------
# CLASSE FCN
#-------------------------------------------------------------------------------

tconv_types = ["8s","16s","32s"]

class FCN(nn.Module):
    
    def __init__(self,in_channels,q,fcn_type="32s"):
        """
            Inicializa uma FCN.

            in_channels: n�mero de canais de entrada
            q: n�mero de classes
            fcn_type: tipo de rede totalmente convolucional.
        """
        super(FCN,self).__init__()

        #-----------------------------------------------------------------------
        # Camadas convolucionais
        #-----------------------------------------------------------------------

        self.conv1_a = nn.Conv2d(in_channels,64,3,padding=100)
        self.relu1_a = nn.ReLU()
        self.conv1_b = nn.Conv2d(64,64,3,padding=1)
        self.relu1_b = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv2_a = nn.Conv2d(64,128,3,padding=1)
        self.relu2_a = nn.ReLU()
        self.conv2_b = nn.Conv2d(128,128,3,padding=1)
        self.relu2_b = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv3_a = nn.Conv2d(128,256,3,padding=1)
        self.relu3_a = nn.ReLU()
        self.conv3_b = nn.Conv2d(256,256,3,padding=1)
        self.relu3_b = nn.ReLU()
        self.conv3_c = nn.Conv2d(256,256,3,padding=1)
        self.relu3_c = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv4_a = nn.Conv2d(256,512,3,padding=1)
        self.relu4_a = nn.ReLU()
        self.conv4_b = nn.Conv2d(512,512,3,padding=1)
        self.relu4_b = nn.ReLU()
        self.conv4_c = nn.Conv2d(512,512,3,padding=1)
        self.relu4_c = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(2,2,ceil_mode=True)

        self.conv5_a = nn.Conv2d(512,512,3,padding=1)
        self.relu5_a = nn.ReLU()
        self.conv5_b = nn.Conv2d(512,512,3,padding=1)
        self.relu5_b = nn.ReLU()
        self.conv5_c = nn.Conv2d(512,512,3,padding=1)
        self.relu5_c = nn.ReLU()
        self.maxpool5 = nn.MaxPool2d(2,2,ceil_mode=True)

        #-----------------------------------------------------------------------
        # Camadas FC s�o substitu�das por camadas convolucionais
        #----------------------------------------------------------------------- 

        self.conv6 = nn.Conv2d(512,4096,7)
        self.relu6 = nn.ReLU()
        self.dropout1 = nn.Dropout2d()

        self.conv7 = nn.Conv2d(4096,4096,1)
        self.relu7 = nn.ReLU()
        self.dropout2 = nn.Dropout2d()

        #-----------------------------------------------------------------------
        # Camadas de convolu��o transposta
        #-----------------------------------------------------------------------

        self.net_type = fcn_type

        if self.net_type == "32s":
            self.score = nn.Conv2d(4096,q,1)
            self.upscore = nn.ConvTranspose2d(q,q,64,stride=32,bias=False)
        
        if self.net_type == "16s":
            self.score = nn.Conv2d(4096,q,1)
            self.score_pool4 = nn.Conv2d(512,q,1)
            self.upscore2 = nn.ConvTranspose2d(q,q,4,2,bias=False)
            self.upscore16 = nn.ConvTranspose2d(q,q,32,16,bias=False)

        if self.net_type == "8s":
            self.score = nn.Conv2d(4096,q,1)
            self.score_pool3 = nn.Conv2d(256,q,1)
            self.score_pool4 = nn.Conv2d(512,q,1)
            self.upscore2 = nn.ConvTranspose2d(q,q,4,2,bias=False)
            self.upscore8 = nn.ConvTranspose2d(q,q,16,8,bias=False)
            self.upscore_pool4 = nn.ConvTranspose2d(q,q,4,2,bias=False)

        #-----------------------------------------------------------------------
        # Inicializa��o dos pesos
        #-----------------------------------------------------------------------
        self.init_weights()
    
    def init_weights(self,from_vgg=True):
        """
            Inicializa pesos da rede.
        """
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                m.weight.data.zero_()
                if m.bias is not None:
                    m.bias.data.zero_()
            if isinstance(m,nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]
                aux = upsampling_weights(m.in_channels,m.out_channels,
                    m.kernel_size[0])
                m.weight.data.copy_(aux)

    def forward(self,x):
        """
            Passa uma entrada x pela rede e retorna um resultado.
        """
        
        h = self.relu1_a(self.conv1_a(x))
        h = self.maxpool1(self.relu2_b(self.conv1_b(h)))

        h = self.relu2_a(self.conv2_a(h))
        h = self.maxpool2(self.relu2_b(self.conv2_b(h)))

        h = self.relu3_a(self.conv3_a(h))
        h = self.relu3_b(self.conv3_b(h))
        h = self.maxpool3(self.relu3_c(self.conv3_c(h)))
        if self.net_type == "8s":
            aux_pool3 = h

        h = self.relu4_a(self.conv4_a(h))
        h = self.relu4_b(self.conv4_b(h))
        h = self.maxpool4(self.relu4_c(self.conv4_c(h)))
        if self.net_type == "16s" or self.net_type == "8s":
            aux_pool4 = h

        h = self.relu5_a(self.conv5_a(h))
        h = self.relu5_b(self.conv5_b(h))
        h = self.maxpool5(self.relu5_c(self.conv5_c(h)))

        h = self.dropout1(self.relu6(self.conv6(h)))

        h = self.dropout2(self.relu7(self.conv7(h)))

        if self.net_type == "32s":
            h = self.score(h)
            h = self.upscore(h)
            return h[:,:,19:19+x.size()[2],19:19+x.size()[3]].contiguous()
        
        if self.net_type == "16s":
            h = self.score(h)
            h = self.upscore2(h)
            aux_upscore2 = h

            h = self.score_pool4(aux_pool4)
            h = h[:,:,5:5+aux_upscore2.size()[2], 5:5+aux_upscore2.size()[3]]
            h = h+aux_upscore2

            h = self.upscore16(h)
            return h[:,:,27:27+x.size()[2],27:27+x.size()[3]].contiguous()
        
        if self.net_type == "8s":
            h = self.score(h)
            h = self.upscore2(h)
            aux_upscore2 = h

            h = self.score_pool4(aux_pool4)
            h = h[:,:,5:5+aux_upscore2.size()[2],5:5+aux_upscore2.size()[3]]
            aux_score4 = h

            h = aux_upscore2+aux_score4
            h = self.upscore_pool4(h)
            aux_upscore4 = h

            h = self.score_pool3(aux_pool3)
            h = h[:,:,9:9+aux_upscore4.size()[2],9:9+aux_upscore4.size()[3]]

            h = aux_upscore4+h

            h = self.upscore8(h)
            return h[:,:,31:31+x.size()[2],31:31+x.size()[3]]
    
    def copy_from_vgg(self,vgg):
        """
            Copia par�metros das camadas de features de uma rede vgg para a 
            camada de features da FCN.
        """
        #Camadas de features da rede atual
        features = [self.conv1_a,self.relu1_a,self.conv1_b,
            self.relu1_b,self.maxpool1,self.conv2_a,self.relu2_a,
            self.conv2_b,self.relu2_b,self.maxpool2,self.conv3_a,
            self.relu3_a,self.conv3_b,self.relu3_b,self.conv3_c,
            self.relu3_c,self.maxpool3,self.conv4_a,self.relu4_a,
            self.conv4_b,self.relu4_b,self.conv4_c,self.relu4_c,
            self.maxpool4,self.conv5_a,self.relu5_a,self.conv5_b,
            self.relu5_b,self.conv5_c,self.relu5_c,self.maxpool5
        ]
        #Copia par�metros da vgg16
        for a,b in zip(features,vgg.features):
            if isinstance(a,nn.Conv2d) and isinstance(b,nn.Conv2d):
                assert a.weight.size() == b.weight.size()
                assert a.bias.size() == b.bias.size()
                a.weight.data = b.weight.data
                a.bias.data = b.bias.data


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# UNET
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

class Block(nn.Module):
    """Fundamental Structure of encoder - decoder architecture

    Args:
        nn (_type_): _description_
    """
    def __init__(self, in_channels, out_channels, kernel_size=(1,3,3)):
        """Initialize a block
        Args:
            in_channels (_type_): Number of the input channels
            out_channels (_type_): Number of the output channels
            kernel_size (_type_): Sizee of the kernel
        """
        super(Block, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.batch1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.batch2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU()
    
    
        # def conv_block(self, in_channels, out_channels):
        # return nn.Sequential(
        #     nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(out_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(out_channels),
        #     nn.ReLU(inplace=True)
        # )
    def forward(self, x):
        conv1 = self.conv1(x)
        batch1 = self.batch1(conv1)
        relu1 = self.relu1(batch1)
        conv2 = self.conv2(relu1)
        batch2 = self.batch2(conv2)
        #relu2 = self.relu2(batch2)
        return self.relu2(batch2)
    
class Encoder(nn.Module):
    """Encoder part of the architecture

    Args:
        nn (_type_): _description_
    """
    def __init__(self,num_filters):
        super(Encoder, self).__init__()
        filters_enc =[3, num_filters * 2, num_filters * 4,
                  num_filters * 8, num_filters * 16]
        self.block = nn.ModuleList(
            [Block(filters_enc[i], filters_enc[i+1]) for i in range(len(filters_enc)-1)]
            )
        self.pool = nn.MaxPool3d((1,2,2))
        self.dropout = nn.Dropout3d(0.1)
        
    def forward(self, x):
        block_out = []
        h = x
        
        for blck in self.block:
            if blck == self.block[-1]:
                h = blck(h)
                block_out.append(h)
            else:
                h = blck(h)
                block_out.append(h)
                h = self.pool(h)
                h = self.dropout(h)   
        return block_out
    
class Decoder(nn.Module):
    def __init__(self, num_filters):
        super(Decoder, self).__init__()
        self.filters_dec = [num_filters * 16, num_filters * 8, num_filters * 4,
                            num_filters * 2]
        self.n_filters_dec = len(self.filters_dec)
        self.tconv = nn.ModuleList(
            [nn.ConvTranspose3d(self.filters_dec[i], self.filters_dec[i+1],(1,3,3),1) for i in range(len(self.filters_dec)-1)]
        )
        self.block = nn.ModuleList(
            [Block(self.filters_dec[i], self.filters_dec[i+1]) for i in range(len(self.filters_dec)-1)]
        )
    
    def crop(self, enc_features, x):
        (_,_,_,H,W) = x.shape
        enc_features = CenterCrop([H,W])(enc_features)
        return enc_features
        
    def forward(self, x, enc_features):
        h = x
        for i in range(self.n_filters_dec-1):
            h = self.tconv[i](h)
            aux_features = self.crop(enc_features[i],h)
            h = torch.cat([h, aux_features], dim=1)
            h = self.block[i](h)
            
        return h
    
    
#-------------------------------------------------------------------------------
# MODELO COMPLETO U-NET
#-------------------------------------------------------------------------------

class UNet(nn.Module):

    def __init__(self, num_filters,n_classes=2):
        super(UNet, self).__init__()
        self.filters_dec = [num_filters * 16, num_filters * 8, num_filters * 4,
                            num_filters * 2]
        self.encoder = Encoder(num_filters)
        self.decoder = Decoder(num_filters)
        self.head = nn.Conv3d(self.filters_dec[-1], n_classes,(1,3,3),1)
    
    def forward(self, x):       
        (_,_,_,H,W) = x.shape
        enc_features = self.encoder(x)
        dec_features = self.decoder(enc_features[::-1][0],
                                    enc_features[::-1][1:])
        mask = self.head(dec_features)
        mask = nn.functional.interpolate(mask,(_,H,W))
        mask = mask.squeeze(2) # Removing an extra dimension for 2D mask
        
        return mask


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# Ellipse RCNN
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

class EllipseRCNN(nn.Module):
    def __init__(self):
        super(EllipseRCNN, self).__init__()
        self.resnet = resnet18(weights=True)
        self.conv1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(1024, 4096)
        self.fc2 = nn.Linear(4096, 2)

        self.ellipse_params = nn.Sequential(
            nn.Linear(2, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 32),
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.conv1(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)

        ellipse_params = self.ellipse_params(x)
        return ellipse_params

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# CASCADE
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# IMPLEMENTA��O DE PEDA�OS DO MODELO -> DECODER
#-------------------------------------------------------------------------------

class conv_block(nn.Module):
    """
        Conv Block do modelo CASCADE. 
        Essa camada � uma parte do Convolutional Attention Module (CAM)
    """
    def __init__(self,in_channels=3):
        """
            Inicializa��o de um conv_block
            in_channels: n�mero de canais de entrada
        """
        super(conv_block,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,in_channels,kernel_size=3)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels,in_channels,kernel_size=3)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu2 = nn.ReLU()
    
    def forward(self,x):
        h = self.relu1(self.bn1(self.conv1(x)))
        h = self.relu2(self.bn2(self.conv2(h)))
        return h

class channel_attention(nn.Module):
    """
        Camada de aten��o do CASCADE.
        Essa camada � uma parte do Convolutional Attention Module (CAM). 
    """

    def __init__(self,in_channels=16):
        """
            Inicializa��o da channel_attention.
            in_channels: n�mero de canais de entrada
        """
        super(channel_attention,self).__init__()
        #Camadas de pooling
        self.aap = nn.AdaptiveAvgPool2d(in_channels)
        self.amp = nn.AdaptiveMaxPool2d(in_channels)
        #Para a sa�da de AAP
        self.conv1_a = nn.Conv2d(in_channels,int(in_channels/16),kernel_size=1)
        self.relu_a = nn.ReLU()
        self.conv2_a = nn.Conv2d(int(in_channels/16),in_channels,kernel_size=1)
        #Para a sa�da de AMP
        self.conv1_b = nn.Conv2d(in_channels,int(in_channels/16),kernel_size=1)
        self.relu_b = nn.ReLU()
        self.conv2_b = nn.Conv2d(int(in_channels/16),in_channels,kernel_size=1)
        #Camada de agrega��o: sigm�ide
        self.sigm = nn.Sigmoid()
    
    def forward(self,x):
        #Passagem do lado AAP
        h_aap = self.aap(x)
        h_aap = self.conv2_a(self.relu_a(self.conv1_a(h_aap)))
        #Passagem do lado AMP
        h_amp = self.amp(x)
        h_amp = self.conv2_b(self.relu_b(self.conv1_b(h_amp)))
        #Agrega��o dos dois lados
        h = self.sigm(h_aap+h_amp)
        return x*h #Retorna produto elemento-por-elemento (Hadamard product)

class spatial_attention(nn.Module):
    """
        Camada de aten��o espacial do CASCADE.
    """
    def __init__(self,in_channels):
        """
            Inicializa uma camada spatial_attention.
            in_channels: n�mero de canais de entrada
        """
        super(spatial_attention,self).__init__()
        self.conv = nn.Conv2d(in_channels,in_channels,kernel_size=7,padding=3)
        self.sigm = nn.Sigmoid()
    
    def forward(self,x):
        h_avg = torch.mean(x,dim=1,keepdim=True)
        h_max, _ = torch.max(x,dim=1,keepdim=True)
        h = torch.cat([h_avg,h_max],dim=1)
        h = self.conv(h)
        return self.sigm(h)*x

class CAM(nn.Module):
    """
        Convolutional Attention Module do CASCADE
    """
    def __init__(self,in_channels):
        super(CAM,self).__init__()
        self.ca = channel_attention(in_channels)
        self.sa = spatial_attention(in_channels)
        self.cblock = conv_block(in_channels)
    
    def forward(self,x):
        return self.cblock(self.sa(self.ca(x)))

class attention_gate(nn.Module):
    """
        Gate de aten��o do CASCADE.
    """
    def __init__(self,in_channels):
        """
            Inicializa attention_gate.
            in_channels: n�mero de canais de entrada.
        """
        super(attention_gate,self).__init__()
        self.conv1_a = nn.Conv2d(in_channels,in_channels,kernel_size=1)
        self.conv1_b = nn.Conv2d(in_channels,in_channels,kernel_size=1)
        self.bn1_a = nn.BatchNorm2d(in_channels)
        self.bn1_b = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels,in_channels,kernel_size=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.sigm = nn.Sigmoid()
    
    def forward(self,x_a,x_b):
        h_a = self.bn1_a(self.conv1_a(x_a))
        h_b = self.bn1_b(self.conv2_b(x_b))
        h = self.relu(h_a+h_b)
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.sigm(h)
        return h*x_b

class upconv(nn.Module):
    """
        UpConv do CASCADE
    """
    def __init__(self,in_channels=3):
        super(upconv,self).__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(in_channels,in_channels,kernel_size=3)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        return self.relu(self.bn(self.conv(self.up(x))))

#-------------------------------------------------------------------------------
# MODELO CASCADE
#-------------------------------------------------------------------------------

class CASCADE(nn.Module):
    """
        Modelo CASCADE
    """
    def __init__(self,encoder,channels):
        super(CASCADE,self).__init__()
        #Decoder pr�-treinado
        self.encoder = encoder
        #Attention gates
        self.ag1 = attention_gate(channels[0])
        self.ag2 = attention_gate(channels[1])
        self.ag3 = attention_gate(channels[2])
        #CAMs
        self.cam1 = CAM(channels[0])
        self.cam2 = CAM(channels[1])
        self.cam3 = CAM(channels[2])
        self.cam4 = CAM(channels[3])
        #Upconvs
        self.up1 = upconv(channels[1])
        self.up2 = upconv(channels[2])
        self.up3 = upconv(channels[3])
        #Convolutions
        self.conv1 = nn.Conv2d(channels[0],channels[0],kernel_size=1)
        self.conv2 = nn.Conv2d(channels[1],channels[1],kernel_size=1)
        self.conv3 = nn.Conv2d(channels[2],channels[2],kernel_size=1)
        self.conv4_a = nn.Conv2d(channels[3],channels[3],kernel_size=1)
        self.conv4_b = nn.Conv2d(channels[3],channels[3],kernel_size=1)
    
    def forward(self,x):
        d1,d2,d3,d4 = self.encoder(x)
        #CASCADE DECODER - NIVEL 4
        h4 = self.cam4(self.conv4_a(d1))
        casc_h4 = self.up3(h4)
        h4 = nn.functional.upsample(self.conv4_b(h4),scale_factor=32)
        #CASCADE DECODER - NIVEL 3
        h3 = torch.cat((self.ag3(d2,casc_h4),casc_h4),dim=1)
        h3 = self.cam3(h3)
        casc_h3 = self.up2(h3)
        h3 = nn.functional.upsample(self.conv3(h3),scale_factor=16)
        #CASCADE DECODER - NIVEL 2
        h2 = torch.cat((self.ag2(d3,casc_h3),casc_h3),dim=1)
        h2 = self.cam2(h2)
        casc_h2 = self.up1(h2)
        h2 = nn.functional.upsample(self.conv2(h2),scale_factor=8)
        #CASCADE DECODER - NIVEL 1
        h1 = torch.cat((self.ag1(d4,casc_h2),casc_h2),dim=1)
        h1 = self.conv1(self.cam1(h1))
        h1 = nn.functional.upsample(h1,scale_factor=4)

        return (h1+h2+h3+h4)/4
    
from pvtv2 import pvt_v2_b2

def PVT_CASCADE(img_size,channels=[512, 320, 128, 64]):
    """
        Retorna o modelo PVT-CASCADE.
        PVT: Pyramid Vision Transformer - Encoder
        CASCADE: Decoder em cascata
    """
    pvt_encoder = pvt_v2_b2(img_size=img_size)  # [64, 128, 320, 512]
    path = 'Modelos/pvt_v2_b2.pth'
    save_model = torch.load(path)
    model_dict = pvt_encoder.state_dict()
    state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    pvt_encoder.load_state_dict(model_dict)

    return CASCADE(pvt_encoder,channels)

