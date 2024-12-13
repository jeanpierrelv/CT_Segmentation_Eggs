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

    def __init__(self,model,optimizer,loss=nn.CrossEntropyLoss(), loss2=nn.MSELoss()):
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
        self.loss2 = loss2.to(self.device)

    def train(self,train_loader,val_loader,n_classes,max_epochs,train_metrics=None,
              val_metrics=None, train_metrics2=None, val_metrics2=None):
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
                
        if train_metrics2:
            tmet2 = [[] for i in range(len(train_metrics2))]
            #Envia métricas para o device
            for f1 in train_metrics2:
                f1.to(self.device)
        if val_metrics2:
            vmet2 = [[] for i in range(len(val_metrics2))]
            #Envia métricas para o device
            for f1 in val_metrics2:
                f1.to(self.device)
        
        #Itera sobre o número total de épocas"
        while n_epoch < max_epochs:

            #---------------------------------------------------------------
            # TREINAMENTO
            #---------------------------------------------------------------

            run_loss = 0
            self.model.train()
            for batch in train_loader:
                #Imagem de entrada e sua máscara de segmentação
                if len(batch) == 3:
                    inputs, masks0, measures = batch
                    measures = measures.to(self.device)
                else:                    
                    inputs, masks0 = batch
                
                # bi = False
                # if type(masks0) is tuple:
                #     masks1, measures = masks0
                #     measures = measures.to(self.device)
                #     bi = True
                # else:
                #     masks1 = masks0.copy()
                #Envia para o device
                inputs = inputs.to(self.device)
                masks0 = masks0.to(self.device)
                # masks1 = masks1.to(self.device)
               
                #Zera gradiente
                self.opt.zero_grad()
                
                # self.model = self.model.to(self.device)
                #Passa inputs pelo modelo
                outs0 = self.model(inputs)
                
                if len(batch) == 3:
                    outs1, outs2 = outs0
                    # outs2.to(self.device)
                else:
                    outs1 = outs0

                # outs1.to(self.device)

                if torch.isnan(outs1).any() or torch.isinf(outs1).any():
                    print("Saída da rede está com problema.")
               
                #Cálculo do custo
                # aux_loss = self.loss(outs1,masks1.long())
                aux_loss = self.loss(outs1,masks0.long())
                if len(batch) == 3:
                    aux_loss2 = self.loss2(outs2, measures.float())
                    a = torch.tensor(1.0, requires_grad=True)
                    aux_total_loss = (1-a)*aux_loss + a*aux_loss2
                    # aux_total_loss = aux_total_loss.float()
                aux_total_loss.backward()
                #Ajusta pesos da rede
                self.opt.step()

                #Armazena custo corrente
                run_loss += aux_total_loss.item()
                #Atualiza métricas de treinamento

                if train_metrics is not None:
                    #Transforma o valor de cada pixel float para uma classe
                    # outs = mask_intervalar(outs,num_classes=3)
                    for f in train_metrics:
                        f.update(outs1,masks0.int())
                
                if len(batch) == 3:
                    if train_metrics2 is not None:
                        #Transforma o valor de cada pixel float para uma classe
                        # outs = mask_intervalar(outs,num_classes=3)
                        for f1 in train_metrics2:
                            f1.update(outs2,measures.float())

    

            #---------------------------------------------------------------
            # VALIDAÇÃO
            #---------------------------------------------------------------

            val_loss = 0
            run_loss_val = 0
            self.model.eval()
            for batch in val_loader:                
                #Imagem de entrada e máscara de segmentação
                if len(batch) == 3:
                    inputs, masks0, measures = batch
                    measures = measures.to(self.device)
                else:                    
                    inputs, masks0 = batch

                # if len(batch) == 3:
                #     masks1, measures = masks0
                #     measures = measures.to(self.device)
                #     bi = True
                # else:
                #     masks1 = masks0.copy()

                #Envia para o device
                inputs = inputs.to(self.device)
                masks0 = masks0.to(self.device)
               
                # self.model = self.model.to(self.device)
                #Passa entrada pelo modelo
                outs0 = self.model.forward(inputs)
                if len(batch) == 3:
                    outs1, outs2 = outs0
                    # outs2.to(self.device)
                else:
                    outs1 = outs0

                # outs1.to(self.device)
                # outs_zero_one = zero_one_encode(outs)  
                # outs_arg = torch.argmax(outs,dim=1)
                
                #Calcula o custo corrente e armazena
                val_loss = self.loss(outs1,masks0.long())
                if len(batch) == 3:
                    val_loss2 = self.loss2(outs2, measures.float())
                    a = torch.tensor(1.0, requires_grad=True)
                    val_total_loss = (1-a)*val_loss + a*val_loss2
                    # val_total_loss = val_total_loss.float()
                #Atualiza métricas de validação
                run_loss_val += val_total_loss.item() 
                if val_metrics is not None:
                    # outs = mask_intervalar(outs,num_classes=3)
                    for f in val_metrics:
                        f.update(outs1,masks0.int())
                
                if len(batch) == 3:
                    if val_metrics2 is not None:
                        #Transforma o valor de cada pixel float para uma classe
                        # outs = mask_intervalar(outs,num_classes=3)
                        for f1 in val_metrics2:
                            f1.update(outs2,measures.float())

            #---------------------------------------------------------------
            # ENCERRAMENTO DE UMA ÉPOCA DE TREINAMENTO
            #---------------------------------------------------------------
            
            #Armazena loss de uma época
            print("Loss de treino: "+str(run_loss))
            tloss.append(run_loss)
            print("Loss de validacao: "+str(run_loss_val))
            vloss.append(run_loss_val)

            #Armazena métricas de uma época e reseta estado interno das mesmas
            if train_metrics:
                for i in range(len(train_metrics)):
                    tmet[i].append(train_metrics[i].compute().item())
                    train_metrics[i].reset()
            if val_metrics:
                for i in range(len(val_metrics)):
                    vmet[i].append(val_metrics[i].compute().item())
                    val_metrics[i].reset()
            if len(batch) == 3:
                if train_metrics2:
                    for i in range(len(train_metrics2)):
                        tmet2[i].append(train_metrics2[i].compute().item())
                        train_metrics2[i].reset()
                if val_metrics2:
                    for i in range(len(val_metrics2)):
                        vmet2[i].append(val_metrics2[i].compute().item())
                        val_metrics2[i].reset()
                
            #Incrementa n_epoch
            print("Terminada época número "+str(n_epoch))
            n_epoch += 1

        to_return = [tloss,vloss]
        if train_metrics:
            to_return.append(tmet)
        if val_metrics:
            to_return.append(vmet)
            
        if train_metrics2:
            to_return.append(tmet2)
        if val_metrics2:
            to_return.append(vmet2)

        return to_return
