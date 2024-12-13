import torch
import torch.nn as nn
import torchmetrics
import numpy as np
from mask_values_intervalar import mask_intervalar, one_hot_3D, zero_one_encode

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

    def train(self,train_loader,val_loader,n_classes,max_epochs,train_metrics=None,
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
        
        #Itera sobre o número total de épocas"
        while n_epoch < max_epochs:

            #---------------------------------------------------------------
            # TREINAMENTO
            #---------------------------------------------------------------

            run_loss = 0
            self.model.train()
            for batch in train_loader:
                #Imagem de entrada e sua máscara de segmentação
                inputs, masks = batch
                # masks_hot = one_hot_3D(masks, num_classes=n_classes)
                # #Normalizacição
                # inputs_max, inputs_min = inputs.max(), inputs.min()
                # masks_max, masks_min = masks.max(), masks.min()
                # inputs = (inputs - inputs_min)/(inputs_max - inputs_min)
                # masks = (masks - masks_min)/(masks_max - masks_min)

                #Envia para o device
                inputs = inputs.to(self.device)
                masks = masks.to(self.device)
                # masks_hot = masks_hot.to(self.device)
                #Zera gradiente
                self.opt.zero_grad()

                #Passa inputs pelo modelo
                outs = self.model(inputs)
                if torch.isnan(outs).any() or torch.isinf(outs).any():
                    print("Saída da rede está com problema.")
                # outs_zero_one = zero_one_encode(outs)    
                # outs_arg = torch.argmax(outs,dim=1)
                #Cálculo do custo
                aux_loss = self.loss(outs,masks.long())
                aux_loss.backward()
                    
                #Ajusta pesos da rede
                self.opt.step()

                #Armazena custo corrente
                run_loss += aux_loss.item()
                #Atualiza métricas de treinamento
                if train_metrics is not None:
                    #Transforma o valor de cada pixel float para uma classe
                    # outs = mask_intervalar(outs,num_classes=3)
                    for f in train_metrics:                     
                        f.update(outs,masks.int())

            #---------------------------------------------------------------
            # VALIDAÇÃO
            #---------------------------------------------------------------

            val_loss = 0
            self.model.eval()
            for batch in val_loader:                
                #Imagem de entrada e máscara de segmentação
                inputs, masks = batch
                # masks_hot = one_hot_3D(masks, num_classes=n_classes)
                
                # #Normalizacição
                # inputs = (inputs - inputs_min)/(inputs_max - inputs_min)
                # masks = (masks - masks_min)/(masks_max - masks_min)

                #Envia para o device
                inputs = inputs.to(self.device)
                masks = masks.to(self.device)
                # masks_hot = masks_hot.to(self.device)
                
                #Passa entrada pelo modelo
                outs = self.model.forward(inputs)
                # outs_zero_one = zero_one_encode(outs)  
                # outs_arg = torch.argmax(outs,dim=1)
                
                #Calcula o custo corrente e armazena
                val_loss += self.loss(outs,masks.long()).item()

                #Atualiza métricas de validação
                if val_metrics is not None:
                    # outs = mask_intervalar(outs,num_classes=3)
                    for f in val_metrics:
                        f.update(outs,masks.int())

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
