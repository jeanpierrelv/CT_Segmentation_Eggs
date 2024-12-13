import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision as tv
from model import UNet, FCN, PVT_CASCADE
from trainer import Trainer
from data_prepare import seg_data
import numpy as np

#-------------------------------------------------------------------------------
# CARREGAMENTO DO CONJUNTO DE DADOS
#-------------------------------------------------------------------------------

train_img = input('Diretório com as imagens de treinamento: ')
train_mask = input('Diretório com as máscaras de treinamento: ')
train_names = np.loadtxt(input('Imagens que serão utilizadas no treinamento: '),
                         dtype=str)
train_data = seg_data(train_names,train_img,train_mask)
train_data.set_crop()

val_img = input('Diretório com as imagens de validação: ')
val_mask = input('Diretório com as máscaras de validação: ')
val_names = np.loadtxt(input('Imagens que serão utilizadas na validação: '),
                       dtype=str)
val_data = seg_data(val_names,val_img,val_mask)
val_data.crop = train_data.crop
val_data.crop_dims = train_data.crop_dims

bs = eval(input('Tamanho dos batches: '))
n_epoch = eval(input('Número de épocas: '))

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# PRIMEIRA BATERIA DE TESTES
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

#-------------------------------------------------------------------------------
# Inicialização do modelo, otimizador e função custo
#-------------------------------------------------------------------------------


gamma_range = [0.001,0.005,0.009,0.01]
p_range = [0.1,0.9]

import torch.optim as topt 
import time

#-------------------------------------------------------------------------------
# Treinamento
#-------------------------------------------------------------------------------

for gamma in gamma_range:
    for p in p_range:
        #Modelo
        model = PVT_CASCADE(train_data.crop_dims)

        #Métrica: índice de Dice e F1Score
        from torchmetrics import Dice, F1Score
        train_metrics = [Dice(),F1Score(task='multiclass',num_classes=2)]
        val_metrics = [Dice(),F1Score(task='multiclass',num_classes=2)]

        #Otimizador
        opt = topt.SGD(model.parameters(),gamma,momentum=p)

        start = time.time()
        trnr = Trainer(model,opt)
        res = trnr.train(train_data.get_loader(bs,True),val_data.get_loader(bs,False),
            n_epoch,train_metrics,val_metrics)
        print('Tempo total de treinamento: ')
        print(time.time()-start)

        #Exporta resultados
        np.savetxt('train_loss_lr='+str(gamma)+',m='+str(p)+'.dat',res[0])
        np.savetxt('val_loss_lr='+str(gamma)+',m='+str(p)+'.dat',res[1])
        np.savetxt('train_met_lr='+str(gamma)+',m='+str(p)+'.dat',res[2])
        np.savetxt('val_met_lr='+str(gamma)+',m='+str(p)+'.dat',res[3])
        torch.save(model.state_dict(),'model_state_lr='+str(gamma)+',m='+str(p)+'.pth')
