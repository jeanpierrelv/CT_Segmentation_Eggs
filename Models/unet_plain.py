import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import CenterCrop
from torchvision.io import read_image
import numpy as np
from os import listdir
import torch.optim as topt 
import time
import torch.nn as nn

#-------------------------------------------------------------------------------    
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------    
#-------------------------------------------------------------------------------
## DATA PREPARE
#-------------------------------------------------------------------------------    
#-------------------------------------------------------------------------------
def unpack_mask(mask: torch.Tensor):
    """
        Transforma uma máscara (h,w) em um 3-tensor (q,h,w), onde q é o número
        de classes. 
    """
    h, w = mask.size() #Dimensões da imagem
    q = torch.max(mask) #Número de classes
    new_mask = torch.zeros((q+1,h,w))

    #Itera sobre pixels da imagem
    for i in range(h):
        for j in range(w):
            s = int(mask[i][j].item())
            new_mask[s][i][j] = 1.
    return new_mask

class seg_data(Dataset):
    """
        Classe para preparação de um dataset para a tarefa de segmentação.
    """
    
    def __init__(self,img_list,img_dir,mask_dir):
        """
            Inicializa um dataset a partir de suas instâncias (x) e resultados
            esperados (targets, y).
            img_list: lista com as imagens a serem utilizadas
            img_dir: diretório com as instâncias
            mask_dir: diretório com as máscaras de segmentação

            Nota: é preciso que os arquivos em x_dir e y_dir tenham os mesmos
            nomes. Isto é, um par (x,y) é identificado pelo mesmo nome, mas em
            diretórios distintos.
        """
        super(seg_data,self).__init__()
        self.img_list = img_list
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        #Propriedades
        self.crop = None
        self.crop_dims = None
        self.mean = None
        self.std = None
        
    def __len__(self):
        return len(self.img_list)

    def set_crop(self):
        """
            Define o par de menores dimensões para realizar a operação
            de CenterCrop.
        """
        min_h = 100000000
        min_w = 100000000
        for aux in self.img_list:
            img = read_image(self.img_dir+'/'+aux)
            c, h, w = img.size()
            if min_h > h:
                min_h = h
            if min_w > w:
                min_w = w
        self.crop = CenterCrop((min_h,min_w))
        self.crop_dims = (min_h,min_w)

    def __getitem__(self,idx):
        if self.crop:
            aux = self.img_list[idx]
            img = read_image(self.img_dir+'/'+aux)
            if self.mean is not None and self.std is not None:
                img = (img.float()-img.min())/(img.max()-img.min())
            img = img.float()
            img = img.unsqueeze(1) # Adding a extra dimension TEST
            aux = aux.split('.')[0]
            mask = torch.load(self.mask_dir+'/'+aux+'.pt').long()
            #mask = unpack_mask(mask).float()
            return self.crop(img), self.crop(mask)
        else:
            raise Exception("self.crop não está definido.")
        
    def get_loader(self,batch_size,shuffle):
        """
            Retorna um dataloader para o conjunto.
        """
        return DataLoader(self,batch_size=batch_size,shuffle=shuffle)
    
    
#-------------------------------------------------------------------------------    
#-------------------------------------------------------------------------------    
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
    def __init__(self, in_channels, out_channels, kernel_size=(3,3,1)):
        """Initialize a block
        Args:
            in_channels (_type_): Number of the input channels
            out_channels (_type_): Number of the output channels
            kernel_size (_type_): Sizee of the kernel
        """
        super(Block, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.batch1 = nn.BatchNorm3d(kernel_size)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.batch2 = nn.BatchNorm3d(kernel_size)
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
    def __init__(self,num_filters=16):
        super(Encoder, self).__init__()
        filters_enc =[num_filters * 1, num_filters * 2, num_filters * 4,
                  num_filters * 8, num_filters * 16]
        self.block = nn.ModuleList(
            [Block(filters_enc[i], filters_enc[i+1]) for i in range(len(filters_enc)-1)]
            )
        self.pool = nn.MaxPool3d(2)
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
        return Block(block_out)
    
class Decoder(nn.Module):
    def __init__(self, num_filters=16):
        super(Decoder, self).__init__()
        filters_dec = [num_filters * 8, num_filters * 4, num_filters * 2,
                            num_filters * 1]
        self.tconv = nn.ModuleList(
            [nn.ConvTranspose3d(filters_dec[i], filters_dec[i+1],3,2) for i in range(len(filters_dec)-1)]
        )
        self.block = nn.ModuleList(
            [Block(filters_dec[i], filters_dec[i+1]) for i in range(len(filters_dec)-1)]
        )
    
    def crop(self, enc_features, x):
        (_,_,H,W) = x.shape
        enc_features = CenterCrop([H,W])(enc_features)
        return enc_features
        
    def forward(self, x, enc_features):
        h = x
        for i in range(len(self.filters_dec)-1):
            h = self.tconv[i]
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
        self.encoder = Encoder(num_filters)
        self.decoder = Decoder(num_filters)
        self.head = nn.Conv3d(num_filters, n_classes,1)
    
    def forward(self, x):
        
        (_,_,_,H,W) = x.shape
        enc_features = self.encoder(x)
        dec_features = self.decoder(enc_features[::-1][0],
                                    enc_features[::-1][0])
        mask = self.head(dec_features)
        mask = nn.functional.interpolate(mask,(H,W))
        
        return mask    


#-------------------------------------------------------------------------------    
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------    
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# TRAINER
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Funções auxiliares
#-------------------------------------------------------------------------------

def get_device(print_device=False):
    """
        Seleciona device disponível. Prioridade: CUDA. Seguido de mps e cpu.
    """
    #Seleciona dispositivo
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if print_device:
            print("torch.device = cuda.")
    else:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            if print_device:
                print("torch.device = mps.")
        else:
            device = torch.device("cpu")
            if print_device:
                print("torch.device = cpu.")
    return device

#-------------------------------------------------------------------------------
# FUNÇÕES PARA O TREINAMENTO (custos e classe Trainer)
#-------------------------------------------------------------------------------
    
class Trainer(object):

    def __init__(self,model,optimizer,loss=nn.CrossEntropyLoss()):
        """
            Classe Trainer para treinar modelos do pytorch.
            
            model: modelo a ser treinado
            optimizer: otimizador a ser utilizado
            loss: função custo. Default: entropia cruzada
        """
        
        self.device = get_device(print_device=True)
        self.model = model.to(self.device)
        self.opt = optimizer
        self.loss = loss.to(self.device)

    def train(self,train_loader,val_loader,max_epochs,train_metrics=None,
              val_metrics=None):
        """
        Função para executar o treinamento de um modelo.

        train_loader: dataloader para o conjunto de treinamento
        val_loader: dataloader para o conjunto de validação
        max_epochs: número máximo de épocas para o treinamento
        
        *Métricas de avaliação do torchmetrics podem ser passadas por meio das 
        listas
        train_metrics: lista de métricas de treinamento
        val_metrics: lista de métricas de validação
        NOTA: Mesmo que sejam passadas as mesmas métricas para treinamento e 
        validação, é necessário passar duas listas distintas por conta do estado
        interno de cada métrica. (Ver documentação do torchmetrics)

        *Sem métricas de avaliação retorna apenas uma lista com o desenvolvimento
        da loss a cada época do treinamento. Quando as métricas são fornecidas,
        retorna-se também as listas contendo essas métricas ao longo do treino e
        da validação.

        """
        
        print("Treinando...")
        #Contador do número de épocas
        n_epoch = 0
        #Lista para armazenar a loss em cada época
        tloss = []
        vloss = []

        #Listas para armazenar as métricas de avaliação a cada época
        if train_metrics:
            tmet = [[] for i in range(len(train_metrics))]
            #Envia métricas para o device
            for f in train_metrics:
                f.to(self.device)
        if val_metrics:
            vmet = [[] for i in range(len(val_metrics))]
            #Envia métricas para o device
            for f in val_metrics:
                f.to(self.device)
        
        #Itera sobre o número total de épocas"O que temos experimentado ao longo dos anos é que, com o aumento do nível do mar, vemos a erosão de partes da ilha."
        while n_epoch < max_epochs:

            #---------------------------------------------------------------
            # TREINAMENTO
            #---------------------------------------------------------------

            run_loss = 0
            self.model.train()
            for batch in train_loader:
                #Imagem de entrada e sua máscara de segmentação
                inputs, masks = batch
                
                #Envia para o device
                inputs = inputs.to(self.device)
                masks = masks.to(self.device)

                #Zera gradiente
                self.opt.zero_grad()

                #Passa inputs pelo modelo
                outs = self.model(inputs)
                if torch.isnan(outs).any() or torch.isinf(outs).any():
                    print("Saída da rede está com problema.")
                #Cálculo do custo
                aux_loss = self.loss(outs,masks)
                aux_loss.backward()
                    
                #Ajusta pesos da rede
                self.opt.step()

                #Armazena custo corrente
                run_loss += aux_loss.item()
                #Atualiza métricas de treinamento
                if train_metrics is not None:
                    for f in train_metrics:
                        f.update(outs,masks)

            #---------------------------------------------------------------
            # VALIDAÇÃO
            #---------------------------------------------------------------

            val_loss = 0
            self.model.eval()
            for batch in val_loader:                
                #Imagem de entrada e máscara de segmentação
                inputs, masks = batch
                
                inputs = inputs.to(self.device)
                masks = masks.to(self.device)

                #Passa entrada pelo modelo
                outs = self.model.forward(inputs)

                #Calcula o custo corrente e armazena
                val_loss += self.loss(outs,masks).item()

                #Atualiza métricas de validação
                if val_metrics is not None:
                    for f in val_metrics:
                        f.update(outs,masks)

            #---------------------------------------------------------------
            # ENCERRAMENTO DE UMA ÉPOCA DE TREINAMENTO
            #---------------------------------------------------------------
            
            #Armazena loss de uma época
            print("Loss de treino: "+str(run_loss))
            tloss.append(run_loss)
            print("Loss de validacao: "+str(val_loss))
            vloss.append(val_loss)

            #Armazena métricas de uma época e reseta estado interno das mesmas
            if train_metrics:
                for i in range(len(train_metrics)):
                    tmet[i].append(train_metrics[i].compute().item())
                    train_metrics[i].reset()
            if val_metrics:
                for i in range(len(val_metrics)):
                    vmet[i].append(val_metrics[i].compute().item())
                    val_metrics[i].reset()
                
            #Incrementa n_epoch
            print("Terminada época número "+str(n_epoch))
            n_epoch += 1

        to_return = [tloss,vloss]
        if train_metrics:
            to_return.append(tmet)
        if val_metrics:
            to_return.append(vmet)

        return to_return
    


#-------------------------------------------------------------------------------    
#-------------------------------------------------------------------------------    
#-------------------------------------------------------------------------------    
#-------------------------------------------------------------------------------
# TEST TRAIN
#-------------------------------------------------------------------------------    
#-------------------------------------------------------------------------------    
    
splits_folder = input('Diretorio com splits: ')

splits = listdir(splits_folder)
img_folder = input('Diretório com as imagens: ')
mask_folder = input('Diretório com as máscaras de segmentação: ')

N_splits = len(splits)/2

bs = eval(input('Tamanho dos batches: '))
n_epoch = eval(input('Número de épocas: '))

#-------------------------------------------------------------------------------
# CARREGAMENTO DO CONJUNTO DE DADOS
#-------------------------------------------------------------------------------
n = 0
while (n<N_splits):

	print('Carregando dados')
	train_split = np.loadtxt(splits_folder+'/train'+str(n)+'.txt',dtype=str)
	train_data = seg_data(train_split,img_folder,mask_folder)
	train_data.set_crop()

	val_split = np.loadtxt(splits_folder+'/train'+str(n)+'.txt',dtype=str)
	val_data = seg_data(val_split,img_folder,mask_folder)
	val_data.crop = train_data.crop
	val_data.crop_dims = train_data.crop_dims

	#-------------------------------------------------------------------------------
	# Inicialização do modelo, otimizador e função custo
	#-------------------------------------------------------------------------------
	
	#Modelo
	model = UNet(num_filters=16)
	
	gamma = 0.001 #learning rate
	p = 0.9 #momentum
	
	#Otimizador
	opt = topt.SGD(model.parameters(),gamma,momentum=p)

	#-------------------------------------------------------------------------------
	# Treinamento
	#-------------------------------------------------------------------------------

        #Métrica: índice de Dice e F1Score
	from torchmetrics import Dice, F1Score
	train_metrics = [Dice(),F1Score(task='multiclass',num_classes=2)]
	val_metrics = [Dice(),F1Score(task='multiclass',num_classes=2)]
	start = time.time()
	trnr = Trainer(model,opt)
	res = trnr.train(train_data.get_loader(bs,True),val_data.get_loader(bs,False),
            n_epoch,train_metrics,val_metrics)
	print('Tempo total de treinamento: ')
	print(time.time()-start)

        #Exporta resultados
	np.savetxt('train_loss_'+str(n)+'.dat',res[0])
	np.savetxt('val_loss_'+str(n)+'.dat',res[1])
	np.savetxt('train_met_'+str(n)+'.dat',res[2])
	np.savetxt('val_met_'+str(n)+'.dat',res[3])
	torch.save(model.state_dict(),'model_state_'+str(n)+'.pth')
	
	n = n+1
