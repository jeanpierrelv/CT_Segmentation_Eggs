import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import CenterCrop
from torchvision.models import resnet18
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torch.nn.functional import grid_sample, interpolate, pad as torch_pad

# from . import _functional_pil as F_pil, _functional_tensor as F_t

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
    
    def __init__(self,in_channels,q,fcn_type="16s"):
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

        self.conv1_a = nn.Conv3d(in_channels,64,3,padding=1)
        self.relu1_a = nn.ReLU()
        self.conv1_b = nn.Conv3d(64,64,3,padding=1)
        self.relu1_b = nn.ReLU()
        self.maxpool1 = nn.MaxPool3d(2,2,ceil_mode=True)

        self.conv2_a = nn.Conv3d(64,128,3,padding=1)
        self.relu2_a = nn.ReLU()
        self.conv2_b = nn.Conv3d(128,128,3,padding=1)
        self.relu2_b = nn.ReLU()
        self.maxpool2 = nn.MaxPool3d(2,2,ceil_mode=True)

        self.conv3_a = nn.Conv3d(128,256,3,padding=1)
        self.relu3_a = nn.ReLU()
        self.conv3_b = nn.Conv3d(256,256,3,padding=1)
        self.relu3_b = nn.ReLU()
        self.conv3_c = nn.Conv3d(256,256,3,padding=1)
        self.relu3_c = nn.ReLU()
        self.maxpool3 = nn.MaxPool3d(2,2,ceil_mode=True)

        self.conv4_a = nn.Conv3d(256,512,3,padding=1)
        self.relu4_a = nn.ReLU()
        self.conv4_b = nn.Conv3d(512,512,3,padding=1)
        self.relu4_b = nn.ReLU()
        self.conv4_c = nn.Conv3d(512,512,3,padding=1)
        self.relu4_c = nn.ReLU()
        self.maxpool4 = nn.MaxPool3d(2,2,ceil_mode=True)

        self.conv5_a = nn.Conv3d(512,512,3,padding=1)
        self.relu5_a = nn.ReLU()
        self.conv5_b = nn.Conv3d(512,512,3,padding=1)
        self.relu5_b = nn.ReLU()
        self.conv5_c = nn.Conv3d(512,512,3,padding=1)
        self.relu5_c = nn.ReLU()
        self.maxpool5 = nn.MaxPool3d(2,2,ceil_mode=True)

        #-----------------------------------------------------------------------
        # Camadas FC s�o substitu�das por camadas convolucionais
        #----------------------------------------------------------------------- 

        self.conv6 = nn.Conv3d(512,4096,2)
        self.relu6 = nn.ReLU()
        self.dropout1 = nn.Dropout3d()

        self.conv7 = nn.Conv3d(4096,4096,1)
        self.relu7 = nn.ReLU()
        self.dropout2 = nn.Dropout3d()

        #-----------------------------------------------------------------------
        # Camadas de convolu��o transposta
        #-----------------------------------------------------------------------

        self.net_type = fcn_type

        if self.net_type == "32s":
            self.score = nn.Conv3d(4096,q,1)
            self.upscore = nn.ConvTranspose3d(q,q,64,stride=32,bias=False)
        
        if self.net_type == "16s":
            self.score = nn.Conv3d(4096,q,1)
            self.score_pool4 = nn.Conv3d(512,q,1)
            self.upscore2 = nn.ConvTranspose3d(q,q,4,2,bias=False)
            self.upscore16 = nn.ConvTranspose3d(q,q,32,16,bias=False)

        if self.net_type == "8s":
            self.score = nn.Conv3d(4096,q,1)
            self.score_pool3 = nn.Conv3d(256,q,1)
            self.score_pool4 = nn.Conv3d(512,q,1)
            self.upscore2 = nn.ConvTranspose3d(q,q,4,2,bias=False)
            self.upscore8 = nn.ConvTranspose3d(q,q,16,8,bias=False)
            self.upscore_pool4 = nn.ConvTranspose3d(q,q,4,2,bias=False)

        #-----------------------------------------------------------------------
        # Inicializa��o dos pesos
        #-----------------------------------------------------------------------
    #     self.init_weights()
    
    # def init_weights(self,from_vgg=False):
    #     """
    #         Inicializa pesos da rede.
    #     """
    #     for m in self.modules():
    #         if isinstance(m,nn.Conv3d):
    #             m.weight.data.zero_()
    #             if m.bias is not None:
    #                 m.bias.data.zero_()
    #         if isinstance(m,nn.ConvTranspose3d):
    #             assert m.kernel_size[0] == m.kernel_size[1]
    #             aux = upsampling_weights(m.in_channels,m.out_channels,
    #                 m.kernel_size[0])
    #             m.weight.data.copy_(aux)

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
            return h[:,:,0:x.size()[2], 0:x.size()[3], 0:x.size()[4]].contiguous()#h[:,:,19:19+x.size()[2],19:19+x.size()[3]].contiguous()
        
        if self.net_type == "16s":
            h = self.score(h)
            h = self.upscore2(h)
            aux_upscore2 = h

            h = self.score_pool4(aux_pool4)
            # h = h[:,:,5:5+aux_upscore2.size()[2], 5:5+aux_upscore2.size()[3]]
            h = h[:,:, 0:min(aux_upscore2.size()[2], h.size()[2]), 0:min(aux_upscore2.size()[3], h.size()[3]), 0:min(aux_upscore2.size()[4], h.size()[4])]
            h = h+aux_upscore2[:,:, 0:min(aux_upscore2.size()[2], h.size()[2]), 0:min(aux_upscore2.size()[3], h.size()[3]), 0:min(aux_upscore2.size()[4], h.size()[4])]

            h = self.upscore16(h)
            return h[:,:, 0:x.size()[2], 0:x.size()[3], 0:x.size()[4]].contiguous()#h[:,:,27:27+x.size()[2],27:27+x.size()[3]].contiguous()
        
        if self.net_type == "8s":
            h = self.score(h)
            h = self.upscore2(h)
            aux_upscore2 = h

            h = self.score_pool4(aux_pool4)
            # h = h[:,:,5:5+aux_upscore2.size()[2],5:5+aux_upscore2.size()[3]]
            h = h[:,:,0:min(aux_upscore2.size()[2], h.size()[2]),
                  0:min(aux_upscore2.size()[3], h.size()[3]),
                  0:min(aux_upscore2.size()[4], h.size()[4])]
            aux_score4 = h

            h = aux_upscore2[:,:,0:min(aux_upscore2.size()[2], h.size()[2]),
                  0:min(aux_upscore2.size()[3], h.size()[3]),
                  0:min(aux_upscore2.size()[4], h.size()[4])]+aux_score4
            h = self.upscore_pool4(h)
            aux_upscore4 = h

            h = self.score_pool3(aux_pool3)
            # h = h[:,:,9:9+aux_upscore4.size()[2],9:9+aux_upscore4.size()[3]]
            h = h[:,:,0:min(aux_upscore4.size()[2],h.size()[2]), 
                         0:min(aux_upscore4.size()[3],h.size()[3]),
                         0:min(aux_upscore4.size()[4],h.size()[4])]

            # h = aux_upscore4+h
            h = aux_upscore4[:,:,0:min(aux_upscore4.size()[2],h.size()[2]), 
                         0:min(aux_upscore4.size()[3],h.size()[3]),
                         0:min(aux_upscore4.size()[4],h.size()[4])] + h

            h = self.upscore8(h)
            return h[:,:,0:x.size()[2],0:x.size()[3], 0:x.size()[4]] #h[:,:,31:31+x.size()[2],31:31+x.size()[3]]
    
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
            if isinstance(a,nn.Conv3d) and isinstance(b,nn.Conv3d):
                assert a.weight.size() == b.weight.size()
                assert a.bias.size() == b.bias.size()
                a.weight.data = b.weight.data
                a.bias.data = b.bias.data


#-------------------------------------------------------------------------------
# CLASSE FCN - Multitask
#-------------------------------------------------------------------------------

tconv_types = ["8s","16s","32s"]

class FCN_Multitask(nn.Module):
    
    def __init__(self,in_channels,q,fcn_type="16s"):
        """
            Inicializa uma FCN.

            in_channels: n�mero de canais de entrada
            q: n�mero de classes
            fcn_type: tipo de rede totalmente convolucional.
        """
        super(FCN_Multitask,self).__init__()

        #-----------------------------------------------------------------------
        # Camadas convolucionais
        #-----------------------------------------------------------------------

        self.conv1_a = nn.Conv3d(in_channels,64,3,padding=1)
        self.relu1_a = nn.ReLU()
        self.conv1_b = nn.Conv3d(64,64,3,padding=1)
        self.relu1_b = nn.ReLU()
        self.maxpool1 = nn.MaxPool3d(2,2,ceil_mode=True)

        self.conv2_a = nn.Conv3d(64,128,3,padding=1)
        self.relu2_a = nn.ReLU()
        self.conv2_b = nn.Conv3d(128,128,3,padding=1)
        self.relu2_b = nn.ReLU()
        self.maxpool2 = nn.MaxPool3d(2,2,ceil_mode=True)

        self.conv3_a = nn.Conv3d(128,256,3,padding=1)
        self.relu3_a = nn.ReLU()
        self.conv3_b = nn.Conv3d(256,256,3,padding=1)
        self.relu3_b = nn.ReLU()
        self.conv3_c = nn.Conv3d(256,256,3,padding=1)
        self.relu3_c = nn.ReLU()
        self.maxpool3 = nn.MaxPool3d(2,2,ceil_mode=True)

        self.conv4_a = nn.Conv3d(256,512,3,padding=1)
        self.relu4_a = nn.ReLU()
        self.conv4_b = nn.Conv3d(512,512,3,padding=1)
        self.relu4_b = nn.ReLU()
        self.conv4_c = nn.Conv3d(512,512,3,padding=1)
        self.relu4_c = nn.ReLU()
        self.maxpool4 = nn.MaxPool3d(2,2,ceil_mode=True)

        self.conv5_a = nn.Conv3d(512,512,3,padding=1)
        self.relu5_a = nn.ReLU()
        self.conv5_b = nn.Conv3d(512,512,3,padding=1)
        self.relu5_b = nn.ReLU()
        self.conv5_c = nn.Conv3d(512,512,3,padding=1)
        self.relu5_c = nn.ReLU()
        self.maxpool5 = nn.MaxPool3d(2,2,ceil_mode=True)

        #-----------------------------------------------------------------------
        # Camadas FC s�o substitu�das por camadas convolucionais
        #----------------------------------------------------------------------- 

        self.conv6 = nn.Conv3d(512,4096,2)
        self.relu6 = nn.ReLU()
        self.dropout1 = nn.Dropout3d()

        self.conv7 = nn.Conv3d(4096,4096,1)
        self.relu7 = nn.ReLU()
        self.dropout2 = nn.Dropout3d()

        #-----------------------------------------------------------------------
        # Camadas de convolu��o transposta
        #-----------------------------------------------------------------------

        self.net_type = fcn_type

        if self.net_type == "32s":
            self.score = nn.Conv3d(4096,q,1)
            self.upscore = nn.ConvTranspose3d(q,q,64,stride=32,bias=False)
        
        if self.net_type == "16s":
            self.score = nn.Conv3d(4096,q,1)
            self.score_pool4 = nn.Conv3d(512,q,1)
            self.upscore2 = nn.ConvTranspose3d(q,q,4,2,bias=False)
            self.upscore16 = nn.ConvTranspose3d(q,q,32,16,bias=False)

        if self.net_type == "8s":
            self.score = nn.Conv3d(4096,q,1)
            self.score_pool3 = nn.Conv3d(256,q,1)
            self.score_pool4 = nn.Conv3d(512,q,1)
            self.upscore2 = nn.ConvTranspose3d(q,q,4,2,bias=False)
            self.upscore8 = nn.ConvTranspose3d(q,q,16,8,bias=False)
            self.upscore_pool4 = nn.ConvTranspose3d(q,q,4,2,bias=False)

        
        self.head_measures = nn.Conv3d(5,1,3,1)#nn.ConvTranspose3d(q,q,32,16,bias=False)
        self.batch_measure = nn.BatchNorm3d(1)
        self.relu = nn.ReLU(inplace=True)
        # self.adapt = nn.AdaptiveAvgPool3d(1) ### added
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc1 = nn.Linear(461472, 70)
        # self.sig = nn.Sigmoid()
        self.fc2 = nn.Linear(70, 3)

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
            return h[:,:,0:x.size()[2], 0:x.size()[3], 0:x.size()[4]].contiguous()#h[:,:,19:19+x.size()[2],19:19+x.size()[3]].contiguous()
        
        if self.net_type == "16s":
            h = self.score(h)
            h = self.upscore2(h)
            aux_upscore2 = h

            h = self.score_pool4(aux_pool4)
            # h = h[:,:,5:5+aux_upscore2.size()[2], 5:5+aux_upscore2.size()[3]]
            h = h[:,:, 0:min(aux_upscore2.size()[2], h.size()[2]), 0:min(aux_upscore2.size()[3], h.size()[3]), 0:min(aux_upscore2.size()[4], h.size()[4])]
            h = h+aux_upscore2[:,:, 0:min(aux_upscore2.size()[2], h.size()[2]), 0:min(aux_upscore2.size()[3], h.size()[3]), 0:min(aux_upscore2.size()[4], h.size()[4])]

            h = self.upscore16(h)

            out = h[:,:, 0:x.size()[2], 0:x.size()[3], 0:x.size()[4]].contiguous()
            measure_pre = self.head_measures(out)
            # measure_pre1 = self.head_measures1(measure_pre)
            measure_batch = self.batch_measure(measure_pre)
            measure_relu = self.relu(measure_batch)
            # measure_adapt = self.adapt(measure_relu)
            measure_flatten = self.flatten(measure_relu)        
            measure_out = self.fc1(measure_flatten)
            # measure_sig = self.sig(measure_out)
            measure_relu1 = self.relu(measure_out)
            measure_out1 = self.fc2(measure_relu1)
            return  out, measure_out1
            #return h[:,:, 0:x.size()[2], 0:x.size()[3], 0:x.size()[4]].contiguous()#h[:,:,27:27+x.size()[2],27:27+x.size()[3]].contiguous()
        
        if self.net_type == "8s":
            h = self.score(h)
            h = self.upscore2(h)
            aux_upscore2 = h

            h = self.score_pool4(aux_pool4)
            # h = h[:,:,5:5+aux_upscore2.size()[2],5:5+aux_upscore2.size()[3]]
            h = h[:,:,0:min(aux_upscore2.size()[2], h.size()[2]),
                  0:min(aux_upscore2.size()[3], h.size()[3]),
                  0:min(aux_upscore2.size()[4], h.size()[4])]
            aux_score4 = h

            h = aux_upscore2[:,:,0:min(aux_upscore2.size()[2], h.size()[2]),
                  0:min(aux_upscore2.size()[3], h.size()[3]),
                  0:min(aux_upscore2.size()[4], h.size()[4])]+aux_score4
            h = self.upscore_pool4(h)
            aux_upscore4 = h

            h = self.score_pool3(aux_pool3)
            # h = h[:,:,9:9+aux_upscore4.size()[2],9:9+aux_upscore4.size()[3]]
            h = h[:,:,0:min(aux_upscore4.size()[2],h.size()[2]), 
                         0:min(aux_upscore4.size()[3],h.size()[3]),
                         0:min(aux_upscore4.size()[4],h.size()[4])]

            # h = aux_upscore4+h
            h = aux_upscore4[:,:,0:min(aux_upscore4.size()[2],h.size()[2]), 
                         0:min(aux_upscore4.size()[3],h.size()[3]),
                         0:min(aux_upscore4.size()[4],h.size()[4])] + h

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
            if isinstance(a,nn.Conv3d) and isinstance(b,nn.Conv3d):
                assert a.weight.size() == b.weight.size()
                assert a.bias.size() == b.bias.size()
                a.weight.data = b.weight.data
                a.bias.data = b.bias.data


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# UNET 3D
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

class Block(nn.Module):
    """Fundamental Structure of encoder - decoder architecture

    Args:
        nn (_type_): _description_
    """
    def __init__(self, in_channels, out_channels, kernel_size=(1,2,2)):
        """Initialize a block
        Args:
            in_channels (_type_): Number of the input channels
            out_channels (_type_): Number of the output channels
            kernel_size (_type_): Sizee of the kernel
        """
        super(Block, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size)
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
        filters_enc =[1, num_filters * 2, num_filters * 4,
                  num_filters * 8, num_filters * 16]
        self.block = nn.ModuleList(
            [Block(filters_enc[i], filters_enc[i+1]) for i in range(len(filters_enc)-1)]
            )
        self.pool = nn.MaxPool3d((2))
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
        self.filters_dec = [num_filters * 16, num_filters * 8, num_filters * 4,num_filters * 2]
        self.n_filters_dec = len(self.filters_dec)
        self.tconv = nn.ModuleList(
            [nn.ConvTranspose3d(self.filters_dec[i], self.filters_dec[i+1],(2,7,7),(2,2,2)) for i in range(len(self.filters_dec)-1)]
        )
        self.block = nn.ModuleList(
            [Block(self.filters_dec[i], self.filters_dec[i+1],1) for i in range(len(self.filters_dec)-1)]
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

    def __init__(self, num_filters,n_classes=5):
        super(UNet, self).__init__()
        self.filters_dec = [num_filters * 16, num_filters * 8, num_filters * 4,
                            num_filters * 2]
        self.encoder = Encoder(num_filters)
        self.decoder = Decoder(num_filters)
        self.head = nn.Conv3d(self.filters_dec[-1], n_classes,1)
    
    def forward(self, x):       
        (_,_,_,H,W) = x.shape
        enc_features = self.encoder(x)
        dec_features = self.decoder(enc_features[::-1][0],
                                    enc_features[::-1][1:])
        mask = self.head(dec_features)
        mask = nn.functional.interpolate(mask,(_,H,W))
        # mask = mask.squeeze(1) # Removing an extra dimension (Depth) for 3D mask
        # sig = nn.Sigmoid()
        # mask = sig(mask)
        # sof = nn.Softmax(dim=1)
        # mask = sof(mask)
        
        
        return mask


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# UNET 3D - MULTITASK
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

class Block_unetm(nn.Module):
    """Fundamental Structure of encoder - decoder architecture

    Args:
        nn (_type_): _description_
    """
    def __init__(self, in_channels, out_channels, kernel_size=(1,2,2)):
        """Initialize a block
        Args:
            in_channels (_type_): Number of the input channels
            out_channels (_type_): Number of the output channels
            kernel_size (_type_): Sizee of the kernel
        """
        super(Block_unetm, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size)
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
    
class Encoder_unetm(nn.Module):
    """Encoder part of the architecture

    Args:
        nn (_type_): _description_
    """
    def __init__(self,num_filters):
        super(Encoder_unetm, self).__init__()
        filters_enc =[1, num_filters * 2, num_filters * 4,
                  num_filters * 8, num_filters * 16]
        self.block = nn.ModuleList(
            [Block_unetm(filters_enc[i], filters_enc[i+1]) for i in range(len(filters_enc)-1)]
            )
        self.pool = nn.MaxPool3d((2))
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
    
class Decoder_unetm(nn.Module):
    def __init__(self, num_filters):
        super(Decoder_unetm, self).__init__()
        self.filters_dec = [num_filters * 16, num_filters * 8, num_filters * 4,num_filters * 2]
        self.n_filters_dec = len(self.filters_dec)
        self.tconv = nn.ModuleList(
            [nn.ConvTranspose3d(self.filters_dec[i], self.filters_dec[i+1],(2,7,7),(2,2,2)) for i in range(len(self.filters_dec)-1)]
        )
        self.block = nn.ModuleList(
            [Block_unetm(self.filters_dec[i], self.filters_dec[i+1],1) for i in range(len(self.filters_dec)-1)]
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
    
# class Measures_unetm(nn.Module):
#     def __init__(self, num_class, out_channels, kernel_size, measurements):
#         super(Measures_unetm, self).__init__()
#         self.conv3d = nn.Conv3d(num_class, out_channels, kernel_size=kernel_size)
#         self.relu = nn.ReLU(inplace=True)
#         self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
#         # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
#         # self.linear = nn.Linear(out_channels, measurements)

#     def forward(self, x, device):
        
#         x = self.conv3d(x)
#         x = self.relu(x)
#         x = self.flatten(x)
#         self.linear = nn.Linear(x.shape[1], 3)
#         x = self.linear(x)
#         return x

#-------------------------------------------------------------------------------
# MODELO COMPLETO U-NET MULTITASK
#-------------------------------------------------------------------------------

class UNet_Multitask(nn.Module):

    def __init__(self, num_filters,n_classes=5):
        super(UNet_Multitask, self).__init__()
        self.filters_dec = [num_filters * 16, num_filters * 8, num_filters * 4,
                            num_filters * 2]
        self.encoder = Encoder_unetm(num_filters)
        self.decoder = Decoder_unetm(num_filters)
        self.head = nn.Conv3d(self.filters_dec[-1], n_classes,1)
        self.head_measures = nn.Conv3d(self.filters_dec[-1], 1, 3, 1)
        self.batch_measure = nn.BatchNorm3d(1)
        self.relu = nn.ReLU(inplace=True)
        # self.adapt = nn.AdaptiveAvgPool3d(1) ### added
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.fc1 = nn.Linear(463334, 70)
        # self.sig = nn.Sigmoid()
        self.fc2 = nn.Linear(70, 3)
        # self.fc3 = nn.Linear(40, 3) ### 463334 ---> 29653376
        # self.measures = Measures_unetm(self.filters_dec[-1], 1, 3s, 1)
    
    def forward(self, x):       
        (_,_,_,H,W) = x.shape
        enc_features = self.encoder(x)
        dec_features = self.decoder(enc_features[::-1][0],
                                    enc_features[::-1][1:])
        mask = self.head(dec_features)
        mask = nn.functional.interpolate(mask,(_,H,W))
        # mask = mask.squeeze(1) # Removing an extra dimension (Depth) for 3D mask
        # sig = nn.Sigmoid()
        # mask = sig(mask)
        # sof = nn.Softmax(dim=1)
        # mask = sof(mask)
        
        # Measurements 
        measure_pre = self.head_measures(dec_features)
        # measure_pre1 = self.head_measures1(measure_pre)
        measure_batch = self.batch_measure(measure_pre)
        measure_relu = self.relu(measure_batch)
        # measure_adapt = self.adapt(measure_relu)
        measure_flatten = self.flatten(measure_relu)        
        measure_out = self.fc1(measure_flatten)
        # measure_sig = self.sig(measure_out)
        measure_relu1 = self.relu(measure_out)
        measure_out1 = self.fc2(measure_relu1)
        # measure_relu1 = self.relu(measure_out1)
        # measure_out2 = self.fc3(measure_relu1)
        # measures_out = self.measures(dec_features)
        
        return mask, measure_out1#, measures_out


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# ConvNeXT
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from functools import partial
# from mmcv_custom import load_checkpoint
# from mmseg.utils import get_root_logger

class Block_convnext(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1) # (N, C, H, W, D) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class ConvNeXt(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=1, num_classes=5, 
                #  depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                #  layer_scale_init_value=1e-6, out_indices=[0, 1, 2, 3],#head_init_scale=1.,
                 depths=[3, 3, 3], dims=[32, 64, 128], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, out_indices=[0, 1, 2],#head_init_scale=1.,
                 ):
        super().__init__()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(len(dims)-1):#3
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)
        
        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(len(dims)):#4
            stage = nn.Sequential(
                *[Block_convnext(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]
        # Upsamples layers ---------------------------
        up_dims=[128, 64, 32]
        self.upsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        up_stem = nn.Sequential(
            nn.ConvTranspose3d(dims[-1], up_dims[0], kernel_size=4, stride=4),
            LayerNorm(up_dims[0], eps=1e-6, data_format="channels_first")
        )
        self.upsample_layers.append(up_stem)
        for i in range(len(up_dims)-1):#3
            upsample_layer = nn.Sequential(
                    LayerNorm(up_dims[i], eps=1e-6, data_format="channels_first"),
                    nn.ConvTranspose3d(up_dims[i], up_dims[i+1], kernel_size=2, stride=2),
            )
            self.upsample_layers.append(upsample_layer)
            
        self.up_stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        up_dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        up_cur = 0
        for i in range(len(up_dims)):#4
            up_stage = nn.Sequential(
                *[Block_convnext(dim=up_dims[i], drop_path=up_dp_rates[up_cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.up_stages.append(up_stage)
            up_cur += depths[i]
        # ---------------------------------------------
        

        # For segmentation ---------------------------
        self.out_indices = out_indices

        norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        for i_layer in range(len(dims)):#4
            layer = norm_layer(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)

        # self.apply(self._init_weights)
        # ---------------------------------------------
        # self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        # self.head = nn.Linear(dims[-1], num_classes)

        # self.apply(self._init_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)
        
        self.pre_head = nn.ConvTranspose3d(up_dims[-1], num_classes, kernel_size=4, stride=4)
        # self.head = nn.Conv3d(num_classes, num_classes, kernel_size=3)
    
    # def crop(self, enc_features, x):
    #     (_,_,_,H,W) = x.shape
    #     enc_features = CenterCrop([H,W])(enc_features)
    #     return enc_features
    
    def precrop(self, img, top, left, height, width, up_deep, depth):
        
        d, h, w = img.shape[2:]
        down_deep = up_deep + depth
        right = left + width
        bottom = top + height
    
        return img[..., up_deep:down_deep, top:bottom, left:right]
    
    def crop_3d(self, img, output_size):
        """Crops the given image at the center.
        If the image is torch Tensor, it is expected
        to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
        If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

        Args:
            img (PIL Image or Tensor): Image to be cropped.
            output_size (sequence or int): (height, width) of the crop box. If int or sequence with single int,
                it is used for both directions.

        Returns:
            PIL Image or Tensor: Cropped image.
        """

        image_depth, image_height, image_width = img.shape[2:]
        crop_depth, crop_height, crop_width = output_size.shape[2:]

        # if crop_depth > image_depth or crop_width > image_width or crop_height > image_height:
        #     padding_ltrb = [
        #         (crop_depth - image_depth) // 2 if crop_depth > image_depth else 0,
        #         (crop_width - image_width) // 2 if crop_width > image_width else 0,
        #         (crop_height - image_height) // 2 if crop_height > image_height else 0,
        #         (crop_depth - image_depth + 1) // 2 if crop_depth > image_depth else 0,
        #         (crop_width - image_width + 1) // 2 if crop_width > image_width else 0,
        #         (crop_height - image_height + 1) // 2 if crop_height > image_height else 0,
        #     ]
        #     # img = pad(img, padding_ltrb, fill=0)  # PIL uses fill value 0
        #     image_depth, image_height, image_width = img.shape[2:]
        #     if crop_width == image_width and crop_height == image_height:
        #         return img
            
        crop_deep = int(round((image_depth - crop_depth) / 2.0))
        crop_top = int(round((image_height - crop_height) / 2.0))
        crop_left = int(round((image_width - crop_width) / 2.0))
        crop_img = self.precrop(img, crop_top, crop_left, crop_height, crop_width, crop_deep, crop_depth)
        return crop_img
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)
        
    def init_weights(self, pretrained=None):  
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            # logger = get_root_logger()
            # load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward_features(self, x):
        down_outs = []
        self.dims=[32, 64, 128]
        for i in range(len(self.dims)):#4
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # return self.norm(x.mean([-2, -1, -3])) # global average pooling, (N, C, H, W, D) -> (N, C)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x_out = norm_layer(x)
                # x_out = LayerNorm(self.dims[i], eps=1e-6, data_format="channels_first")(x)#norm_layer(x)
                down_outs.append(x_out)

        # return tuple(outs)
        # Upsamples layers ---------------------------
        up_outs = []
        self.up_dims=[128, 64, 32]
        for i in range(len(self.up_dims)):#4
            if i == 0:
                x = self.upsample_layers[i](down_outs[-1])
            else:
                x = self.upsample_layers[i](x)
            x = self.up_stages[i](x)
        # return self.norm(x.mean([-2, -1, -3])) # global average pooling, (N, C, H, W, D) -> (N, C)
            if i in self.out_indices:
                # norm_layer = getattr(self, f'norm{i}')
                # x_up_out = LayerNorm(self.up_dims[i], eps=1e-6, data_format="channels_first")(x)#norm_layer(x)
                # aux_features = self.crop(x_up_out, outs[3-i])
                aux_features = self.crop_3d(x, down_outs[len(self.up_dims)-1-i])
                # x_up_out = torch.cat([x_up_out, aux_features], dim=1)
                x_up_out = torch.cat([aux_features, down_outs[len(self.up_dims)-1-i]], dim=1)
                x_up_out = nn.Conv3d(x_up_out.shape[1], aux_features.shape[1], kernel_size=1)(x_up_out)#, device=self.device
                up_outs.append(x_up_out)

        return up_outs[-1]
        
    def forward(self, x):
        D,H,W = x.shape[2:]
        x = self.forward_features(x)
        x = self.pre_head(x)
        x = nn.functional.interpolate(x,(D,H,W))
        # x = self.head(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")   
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = ((x - u) / torch.sqrt(s + self.eps))
            x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            return x


# class LayerNorm(nn.Module):
#     r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
#     The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
#     shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
#     with shape (batch_size, channels, height, width).
#     """
#     def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last", device=None):
#         super().__init__()
#         self.normalized_shape = (normalized_shape,)
#         # self.normalized_shape = normalized_shape
#         self.eps = eps
#         self.data_format = data_format

#         if self.data_format not in ["channels_last", "channels_first"]:
#             raise NotImplementedError

#         # Use provided device or default to CPU
#         self.device = device or torch.device("cpu")

#         self.weight = nn.Parameter(torch.ones(normalized_shape, device=self.device))
#         self.bias = nn.Parameter(torch.zeros(normalized_shape, device=self.device))

#     def forward(self, x):
#         if self.data_format == "channels_last":
#             return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
#         else:
#             # Leverage F.layer_norm with appropriate dim specification
#             x1 = x.view(-1, self.normalized_shape[0]).to(device=self.device)
#             # out = F.layer_norm(x1, self.normalized_shape, weight=self.weight.unsqueeze(1), bias=self.bias.unsqueeze(1), eps=self.eps)
#             out = F.layer_norm(x1, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)
#             return out.reshape((x.shape))

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# MASK-RCNN
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

class MaskRCNN(nn.Module):
    def __init__(self, num_classes):
        super(MaskRCNN, self).__init__()
        # Load pre-trained Mask R-CNN model (ResNet-50 FPN)
        self.model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
        # Modify the output layer to match your number of classes
        # self.model.roi_heads.box_predictor.linear.out_features = num_classes

    def forward(self, images):
        # Pass images through the model
        predictions = self.model(images)
        # Extract masks from the predictions
        masks = predictions["masks"]
        return masks

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

