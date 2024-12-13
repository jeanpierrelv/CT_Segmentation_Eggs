from os import listdir

img_dir = input('Diretório com as imagens: ')
split_ratio = eval(input('Taxa de imagens no conjunto de treinamento: '))
n_splits = eval(input('Número de splits a serem feitos: '))
splits_dir = input('Diretório para salvar os splits: ')
                      
#Número total de imagens
N = len(listdir(img_dir))
#Número de imagens no conjunto de treinamento
N_train = int(split_ratio*N)
#Número de imagens no conjunto de validação
N_val = (N-N_train)

import numpy as np

for n in range(n_splits):
    test_imgs = []
    val_imgs = [] 
    train_imgs = listdir(img_dir)
    for r in range(N_val):
        idx = np.random.randint(0,len(train_imgs))
        val_imgs.append(train_imgs.pop(idx))
    #Número de imagens no conjunto de teste
    N_test = N-N_train-int((N-N_train)*0.5)
    for r in range(N_test):
        idx = np.random.randint(0,len(val_imgs))
        test_imgs.append(val_imgs.pop(idx))
    np.savetxt(splits_dir + '/' + 'train'+str(n)+'.txt',train_imgs,fmt="%s")
    np.savetxt(splits_dir + '/' + 'val'+str(n)+'.txt',val_imgs,fmt="%s")
    np.savetxt(splits_dir + '/' + 'test'+str(n)+'.txt',test_imgs,fmt="%s")
