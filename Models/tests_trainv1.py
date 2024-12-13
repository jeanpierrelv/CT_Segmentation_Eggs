import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision as tv
from model_3D import UNet, FCN, MaskRCNN, UNet_Multitask, FCN_Multitask#, PVT_CASCADE, EllipseRCNN
from trainer import Trainer
from data_prepare_3D import seg_data
import numpy as np
import torch.optim as topt 
import time
from os import listdir, path
import matplotlib.pyplot as plt
from torchmetrics import Dice, F1Score, Accuracy, CohenKappa, MatthewsCorrCoef, HammingDistance, MeanSquaredError, MeanAbsoluteError
from torchmetrics.classification import MulticlassF1Score, MulticlassCohenKappa, MulticlassMatthewsCorrCoef, AUROC,MulticlassHammingDistance

splits_folder = input('Diretorio com splits: ')
splits =[]
for file in listdir(splits_folder):
    if file.endswith(".txt"):
        splits.append(file)
        
# splits = listdir(splits_folder)
img_folder = input('Diretório com as imagens: ')
mask_folder = input('Diretório com as máscaras de segmentação: ')

N_splits = len(splits)/3

bs = eval(input('Tamanho dos batches: '))
n_epoch = eval(input('Número de épocas: '))
n_classes = eval(input('Número de clases: '))

#-------------------------------------------------------------------------------
# CARREGAMENTO DO CONJUNTO DE DADOS
#-------------------------------------------------------------------------------
n = 0
while (n<N_splits):

	print('Carregando dados')
	train_split = np.loadtxt(splits_folder+'/train'+str(n)+'.txt',dtype=str)
	train_data = seg_data(train_split,img_folder,mask_folder)
	train_data.set_crop()

	val_split = np.loadtxt(splits_folder+'/val'+str(n)+'.txt',dtype=str)
	val_data = seg_data(val_split,img_folder,mask_folder)
	val_data.crop = train_data.crop
	val_data.crop_dims = train_data.crop_dims

	#-------------------------------------------------------------------------------
	# Inicialização do modelo, otimizador e função custo
	#-------------------------------------------------------------------------------

	#Modelo
	model = UNet_Multitask(num_filters=8, n_classes=n_classes)
	# model = FCN_Multitask(1,5)
    # model = UNet(num_filters=8, n_classes=n_classes)  
	# model = FCN(1,5)
	# model = ConvNeXt()
	
	# model = MaskRCNN(5)
	transfer_learning0 = False
	if transfer_learning0:
		model_path = 'results/unet/split_80_10_10/epochs_150/model_state_0.pth'
		# model_path = 'results/fcn/split_80_10_10/epochs_1000_lr_00001/model_state_0.pth'
		checkpoint = torch.load(model_path)
		model.load_state_dict(checkpoint)

	second_stage = True
	transfer_learning = True
	if transfer_learning:
# 		model_path = 'results/unet/split_80_10_10/epochs_150/model_state_0.pth'
		# model_path = 'data/data_backup/real_data_dicom/mix_filter_data/results_filter_mix/fcn/segmentation_only/model_state_0.pth'
		model_path = 'data/data_backup/real_data_dicom/mix_filter_data/model_state_0.pth'
		checkpoint = torch.load(model_path)
		model.load_state_dict(checkpoint,strict=False)
		layers_to_finetune = [
            'head_measures.weight',
            'head_measures.bias',
            'fc1.weight', 'fc1.bias',
			'fc2.weight', 'fc2.bias',
            ]
        # Freeze all layers except the specified ones
		for name, param in model.named_parameters():
			if name not in layers_to_finetune:
				param.requires_grad = False
     
	lr = 0.00001 #learning rate = 0.00001(unet-mse)
	p = 0.9 #momentum
	
	#Otimizador
	# opt = topt.SGD(model.parameters(),lr,momentum=p)
	opt = topt.Adam(model.parameters(),lr)
	#-------------------------------------------------------------------------------
	# Treinamento
	#-------------------------------------------------------------------------------
    #Métrica: índice de Dice e F1Score
	train_metrics = [Accuracy(task="multiclass", num_classes=n_classes),
                  	MulticlassF1Score(num_classes=n_classes),
                       #AUROC(task="multiclass", num_classes=n_classes),
                  	MulticlassCohenKappa(num_classes=n_classes),
                    MulticlassMatthewsCorrCoef(num_classes=n_classes),
                         MulticlassHammingDistance(num_classes=n_classes)]
	# train_metrics2 = [MeanSquaredError(), MeanAbsoluteError()]
    
	val_metrics = [Accuracy(task="multiclass", num_classes=n_classes),
                  	MulticlassF1Score(num_classes=n_classes),
                       #AUROC(task="multiclass", num_classes=n_classes),
                  	MulticlassCohenKappa(num_classes=n_classes),
                    MulticlassMatthewsCorrCoef(num_classes=n_classes),
                         MulticlassHammingDistance(num_classes=n_classes)]
    
	if second_stage == True:
		train_metrics2 = [MeanSquaredError(), MeanAbsoluteError()]
		val_metrics2 = [MeanSquaredError(), MeanAbsoluteError()]
	start = time.time()
	trnr = Trainer(model,opt)
	if second_stage == False:
		res = trnr.train(train_data.get_loader(bs,True),val_data.get_loader(bs,False),
				n_classes,n_epoch,train_metrics,val_metrics)#, train_metrics2, val_metrics2)
	elif second_stage == True:
		res = trnr.train(train_data.get_loader(bs,True),val_data.get_loader(bs,False),
				n_classes,n_epoch,train_metrics,val_metrics, train_metrics2, val_metrics2)


	# val_metrics2 = [MeanSquaredError(), MeanAbsoluteError()]
	# start = time.time()
	# trnr = Trainer(model,opt)
	# res = trnr.train(train_data.get_loader(bs,True),val_data.get_loader(bs,False),
    #         n_classes,n_epoch,train_metrics,val_metrics, train_metrics2, val_metrics2)
	print('Tempo total de treinamento: ')
	print(time.time()-start)

        #Exporta resultados
	np.savetxt(splits_folder + '/' + 'train_loss_'+str(n)+'.dat',res[0])
	np.savetxt(splits_folder + '/' + 'val_loss_'+str(n)+'.dat',res[1])
	np.savetxt(splits_folder + '/' + 'train_met_'+str(n)+'.dat',res[2])
	np.savetxt(splits_folder+ '/' + 'val_met_'+str(n)+'.dat',res[3])
	if second_stage == True:
		np.savetxt(splits_folder + '/' + 'train_met1_'+str(n)+'.dat',res[4])
		np.savetxt(splits_folder+ '/' + 'val_met1_'+str(n)+'.dat',res[5])
	# np.savetxt(splits_folder + '/' + 'train_met1_'+str(n)+'.dat',res[4])
	# np.savetxt(splits_folder+ '/' + 'val_met1_'+str(n)+'.dat',res[5])
	torch.save(model.state_dict(),splits_folder + '/' + 'model_state_'+str(n)+'.pth')
	
	n = n+1


# num_examples = 2  # Adjust the number of examples to display
# model.eval()


# fig, axes = plt.subplots(num_examples, 3, figsize=(12, num_examples * 4))

# for i in range(num_examples=2):
#     outs = model(val_data.get_loader(bs,False).dataset[i][0])
# 	# Plot the original image
# 	axes[i, 0].imshow(np.transpose(val_data.get_loader(bs,False).dataset[i][0], (1, 2, 0)), cmap='gray')
# 	axes[i, 0].set_title('Original Image')
# 	axes[i, 0].axis('off')

# 	# Plot the ground truth mask
# 	axes[i, 1].imshow(val_data.get_loader(bs,False).dataset[i][1][i, 0], cmap='viridis')  # Assuming binary masks
# 	axes[i, 1].set_title('Ground Truth Mask')
# 	axes[i, 1].axis('off')

# 	# Plot the predicted mask
# 	axes[i, 2].imshow(outs[i, 0], cmap='viridis')  # Assuming binary masks
# 	axes[i, 2].set_title('Predicted Mask')
# 	axes[i, 2].axis('off')

# plt.tight_layout()
# plt.show()