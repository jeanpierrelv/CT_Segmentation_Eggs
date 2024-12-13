from os import listdir, mkdir
import skimage as ski
import numpy as np

#Carrega pastas com imagens
img_dir = input("Diretório com as imagens: ")
mask_dir = input("Diretório com as máscaras de segmentação: ")

#Lista das imagens (as imagens em img_dir e maxk_dir precisam ter o mesmo nome)
img_list = listdir(img_dir)

mkdir('aug')
mkdir('aug/images')
mkdir('aug/masks')

#Itera sobre as imagens
for aux in img_list:
    aux_split = aux.split('.')
    #Carrega imagem e sua respectiva máscara de segmetação
    img = ski.io.imread(img_dir+"/"+aux)
    mask = ski.io.imread(mask_dir+"/"+aux)
    #Primeira transformação
    aux_img = np.flipud(img)
    aux_mask = np.flipud(mask)
    ski.io.imsave("aug/images/"+aux_split[0]+"_a."+aux_split[1],aux_img)
    ski.io.imsave("aug/masks/"+aux_split[0]+"_a."+aux_split[1],aux_mask)
    #Segunda transformação
    aux_img = np.fliplr(aux_img)
    aux_mask = np.fliplr(aux_mask)
    ski.io.imsave("aug/images/"+aux_split[0]+"_b."+aux_split[1],aux_img)
    ski.io.imsave("aug/masks/"+aux_split[0]+"_b."+aux_split[1],aux_mask)
    #Terceira transformação
    aux_img = np.fliplr(img)
    aux_mask = np.fliplr(mask)
    ski.io.imsave("aug/images/"+aux_split[0]+"_c."+aux_split[1],aux_img)
    ski.io.imsave("aug/masks/"+aux_split[0]+"_c."+aux_split[1],aux_mask)