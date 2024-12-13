import torch
import torch.nn as nn
from torchvision.transforms import CenterCrop

# in_channels=3
# out_channels=16
# kernel_size=3

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
        self.filters_enc =[3, num_filters * 2, num_filters * 4,
                  num_filters * 8, num_filters * 16]
        self.block = nn.ModuleList(
            [Block(self.filters_enc[i], self.filters_enc[i+1]) for i in range(len(self.filters_enc)-1)]
            )
        self.pool = nn.MaxPool3d((1,2,2))
        self.dropout = nn.Dropout3d(0.1)
        
    def forward(self, x):
        block_out = []
        h = x
        
        for blck in self.block:
            h = blck(h)
            block_out.append(h)
            if blck == self.block[-1]:
                pass
            else:
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
class YourModel(nn.Module):
    def __init__(self):
        super(YourModel, self).__init__()
        self.conv_layer = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.batch_norm = nn.BatchNorm3d(out_channels)  # Number of channels after conv_layer
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size)
        self.batch2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.batch_norm(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.relu2(x)
        return x

class UNet(nn.Module):

    def __init__(self, num_filters,n_classes=2):
        super(UNet, self).__init__()
        self.filters_dec = [num_filters * 16, num_filters * 8, num_filters * 4,
                            num_filters * 2]
        self.encoder = Encoder(num_filters)
        self.decoder = Decoder(num_filters)
        self.head = nn.Conv3d(self.filters_dec[-1], n_classes,3,1)
    
    def forward(self, x):       
        (_,_,_,H,W) = x.shape
        enc_features = self.encoder(x)
        dec_features = self.decoder(enc_features[::-1][0],
                                    enc_features[::-1][1:])
        mask = self.head(dec_features)
        mask = nn.functional.interpolate(mask,(H,W))
        
        return mask
    
    
# Instantiate and test the model
model1 = Encoder(num_filters=16)
decoder = Decoder(num_filters=16)
input_data = torch.randn(4, 3, 1, 352, 332)  # Example input data
output1 = model1(input_data)

output2 = decoder(output1[::-1][0], output1[::-1][1:])

head = nn.Conv3d(32, 2,(1,3,3),1)

mask = head(output2)

(_,_,_,H,W) = input_data.shape
mask1 = nn.functional.interpolate(mask,(_,H,W))

for i in range(len(output1)):
    print(f"Output shape: {i}", output1[i].shape)
    
print(f"Mask shape: {i}", mask.shape)

print(f"Mask1 shape: {i}", mask1.shape)

tensor_with_grad = mask1[0][0][0]
numpy_array = tensor_with_grad.detach().numpy()
