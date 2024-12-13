import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# # Sample data (replace with your loaded data)
# train_dice = np.array([0.85, 0.90])
# val_dice = np.array([0.82, 0.88])
# train_f1 = np.array([0.78, 0.85])
# val_f1 = np.array([0.75, 0.83])
# train_loss = np.array([0.1, 0.08])
# val_loss = np.array([0.15, 0.12])

# folder_path = 'results/unet/split_80_10_10/epochs_100'
# #'results/fcn/results_20240312_FCN_1pad_2conv6_1000epochs_10batch_lr00001_16s'
# #'results/results_simulate_231023_100epochs_adam_0001_out3_5class_nonorm_with_metrics'
# train_metrics = np.loadtxt(folder_path + '/' +'train_met_0.dat')
# train_loss = np.loadtxt(folder_path + '/' +'train_loss_0.dat')
# val_metrics = np.loadtxt(folder_path + '/' +'val_met_0.dat')
# val_loss = np.loadtxt(folder_path + '/' +'val_loss_0.dat')


models = eval(input('Select the group of models compare: unet(1), fcn(2), maskrcnn(3), convnext(4): '))
len_models = len(models)
folder_path = 'data/data_backup/real_data_dicom/mix_filter_data'#results/mlp'
#-------------------------------------------------------------------------------
# Models
#-------------------------------------------------------------------------------

models_compare = []
train_metrics = []
val_metrics = []
train_loss = []
val_loss = []
# UNET
if 1 in models:
    model_name = "unet"
    models_compare.append(model_name)
    train_metrics.append(np.loadtxt(folder_path + '/' + f'train_met_0_{model_name}.dat'))
    val_metrics.append(np.loadtxt(folder_path + '/' + f'val_met_0_{model_name}.dat'))
    train_loss.append(np.loadtxt(folder_path + '/' + f'train_loss_0_{model_name}.dat'))
    val_loss.append(np.loadtxt(folder_path + '/' +f'val_loss_0_{model_name}.dat'))
# FCN
if 2 in models:
    model_name = "fcn"
    models_compare.append(model_name)
    train_metrics.append(np.loadtxt(folder_path + '/' + f'train_met_0_{model_name}.dat'))
    val_metrics.append(np.loadtxt(folder_path + '/' + f'val_met_0_{model_name}.dat'))
    train_loss.append(np.loadtxt(folder_path + '/' + f'train_loss_0_{model_name}.dat'))
    val_loss.append(np.loadtxt(folder_path + '/' +f'val_loss_0_{model_name}.dat'))





fig = plt.plot(figsize=(4, 4))
for i in range(0, len_models):
    g = sns.lineplot(train_metrics[i][0], lw=3, label=f'{round(train_metrics[i][0][-1],4)} {models_compare[i]} Accuracy - Train')
    sns.lineplot(val_metrics[i][0], lw=2, label=f'{round(val_metrics[i][0][-1],4)} {models_compare[i]} Accuracy - Valid')
    sns.lineplot(train_metrics[i][1], lw=3, label=f'{round(train_metrics[i][1][-1],4)} {models_compare[i]} F1-Score - Train')
    sns.lineplot(val_metrics[i][1], lw=2, label=f'{round(val_metrics[i][1][-1],4)} {models_compare[i]} F1-Score - Valid')
    sns.lineplot(train_metrics[i][2], lw=3, label=f'{round(train_metrics[i][2][-1],4)} {models_compare[i]} Kappa Score - Train')
    sns.lineplot(val_metrics[i][2], lw=2,label=f'{round(val_metrics[i][2][-1],4)} {models_compare[i]} Kappa Score - Valid')
    sns.lineplot(train_metrics[i][3], lw=3, label=f'{round(train_metrics[i][3][-1],4)} {models_compare[i]} Matthew Correlation - Train')
    sns.lineplot(val_metrics[i][3], lw=2,label=f'{round(val_metrics[i][3][-1],4)} {models_compare[i]} Matthew Score - Valid')
plt.xlabel('Epochs', size=18, fontstyle='italic', weight=900)
plt.ylabel('Metrics', size=18, fontstyle='italic', weight=900)
plt.setp(g.get_legend().get_texts(), fontsize='16')   
plt.tight_layout()
plt.show()

fig = plt.plot(figsize=(4, 4))
for i in range(0, len_models):
    g = sns.lineplot(train_loss[i], lw=3, label = f'{round(train_loss[i][-1],4)} {models_compare[i]} Loss - Train')
    sns.lineplot(val_loss[i], lw=2, label = f'{round(val_loss[i][-1],4)} {models_compare[i]} Loss - Valid')
    # plt.legend(labels = [f'{round(train_loss[-1],4)}  Loss - Training', f'{round(val_loss[-1],4)}  Loss - Validation'])
plt.xlabel('Epochs', size=18, fontstyle='italic', weight=900)
plt.ylabel('Loss', size=18, fontstyle='italic', weight=900)
plt.setp(g.get_legend().get_texts(), fontsize='16')  
plt.tight_layout()
plt.show()










# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# fig.legend(ncol=2)

# ax1.plot(train_metrics[0], label=f'{round(train_metrics[0][-1],4)}  ACCURACY - Training', linewidth=2)
# ax1.plot(val_metrics[0], label=f'{round(val_metrics[0][-1],4)}  ACCURACY - Validation', linewidth=0.7)
# ax1.plot(train_metrics[1], label=f'{round(train_metrics[1][-1],4)}  F1-SCORE - Training', linewidth=2)
# ax1.plot(val_metrics[1], label=f'{round(val_metrics[1][-1],4)}  F1-SCORE - Validation', linewidth=0.7)
# # ax1.plot(train_metrics[2], label='AUROC - Training', linewidth=2)
# # ax1.plot(val_metrics[2], label='AUROC - Validation', linewidth=0.7)
# ax1.plot(train_metrics[2], label=f'{round(train_metrics[2][-1],4)}  MATTHEW CORRELATION - Training', linewidth=2)
# ax1.plot(val_metrics[2], label=f'{round(val_metrics[2][-1],4)}  MATTHEW CORRELATION - Validation', linewidth=0.7)
# ax1.plot(train_metrics[3], label=f'{round(train_metrics[3][-1],4)}  KAPPA SCORE - Training', linewidth=2)
# ax1.plot(val_metrics[3], label=f'{round(val_metrics[3][-1],4)}  KAPPA SCORE - Validation', linewidth=0.7)
# # ax1.plot(train_metrics[4], label='HAMMING DISTANCE - Training', linewidth=2)
# # ax1.plot(val_metrics[4], label='HAMMING DISTANCE - Validation', linewidth=0.7)
# ax1.set_ylabel('Scores')
# ax1.set_title('DICE and F1 Score')
# # ax1.set_xticks(x)
# # ax1.set_xticklabels(labels)
# ax1.legend()
# ax2.plot(train_loss, label=f'{round(train_loss[-1],4)}  Loss - Training')
# ax2.plot(val_loss, label=f'{round(val_loss[-1],4)}  Loss - Validation')
# ax2.set_ylabel('Loss')
# ax2.set_title('Loss')
# ax2.legend()
# plt.tight_layout()
# plt.show()
# # Optionally, save the plot to a file
# # plt.savefig('metrics_and_loss.png')

# # sns.set_style('darkgrid')
# # sns.set(rc={'figure.figsize':(14,8)})
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# sns.lineplot(train_metrics[0], lw=3, ax=axes[0], label=f'{round(train_metrics[0][-1],4)}  ACCURACY - Training')
# sns.lineplot(val_metrics[0], lw=2, ax=axes[0], label=f'{round(val_metrics[0][-1],4)}  ACCURACY - Validation')
# sns.lineplot(train_metrics[1], lw=3, ax=axes[0], label=f'{round(train_metrics[1][-1],4)}  F1-SCORE - Training')
# sns.lineplot(val_metrics[1], lw=2, ax=axes[0], label=f'{round(val_metrics[1][-1],4)}  F1-SCORE - Validation')
# # sns.lineplot(train_metrics[2],  palette='viridis', lw=3, ax=axes[0], label='AUROC - Training')
# # sns.lineplot(val_metrics[2],  palette='viridis', lw=2, ax=axes[0], label='AUROC - Validation')
# sns.lineplot(train_metrics[2], lw=3, ax=axes[0], label=f'{round(train_metrics[2][-1],4)}  MATTHEW CORRELATION - Training')
# sns.lineplot(val_metrics[2], lw=2, ax=axes[0], label=f'{round(val_metrics[2][-1],4)}  MATTHEW CORRELATION - Validation')
# sns.lineplot(train_metrics[3], lw=3, ax=axes[0], label=f'{round(train_metrics[3][-1],4)}  KAPPA SCORE - Training')
# sns.lineplot(val_metrics[3], lw=2, ax=axes[0], label=f'{round(val_metrics[3][-1],4)}  KAPPA SCORE - Validation')
# axes[0].legend(fontsize=12)
# axes[0].set_xlabel('Epochs', size=16, fontstyle='italic', weight=900)
# axes[0].set_ylabel('Accuracy', size=16, fontstyle='italic', weight=900)
# # sns.lineplot(train_metrics[4],  palette='viridis', lw=3, ax=axes[0], label='HAMMING DISTANCE - Training')
# # sns.lineplot(val_metrics[4],  palette='viridis', lw=2, ax=axes[0], label='HAMMING DISTANCE - Validation')
# # plt.legend(labels = ['DICE - Training', 'DICE - Validation', 'F1-SCORE - Training', 'F1-SCORE - Validation'])
# sns.lineplot(train_loss, lw=3, ax=axes[1])
# sns.lineplot(val_loss, lw=2, ax=axes[1])
# plt.legend(labels = [f'{round(train_loss[-1],4)}  Loss - Training', f'{round(val_loss[-1],4)}  Loss - Validation'], fontsize=14)
# axes[1].set_xlabel('Epochs', size=16, fontstyle='italic', weight=900)
# axes[1].set_ylabel('Loss', size=16, fontstyle='italic', weight=900)
# # plt.xlabel('Epochs', size=16, fontstyle='italic', weight=900, ax=axes[1])
# # sns.set_context(x)   ylabel('', size=16, fontstyle='italic', weight=900)
# plt.tight_layout()
# plt.show()


# fig = plt.plot(figsize=(4, 4))
# sns.lineplot(train_metrics[0], lw=3, label=f'{round(train_metrics[0][-1],4)}  ACCURACY - Training')
# sns.lineplot(val_metrics[0], lw=2, label=f'{round(val_metrics[0][-1],4)}  ACCURACY - Validation')
# sns.lineplot(train_metrics[1], lw=3, label=f'{round(train_metrics[1][-1],4)}  F1-SCORE - Training')
# sns.lineplot(val_metrics[1], lw=2, label=f'{round(val_metrics[1][-1],4)}  F1-SCORE - Validation')
# sns.lineplot(train_metrics[2], lw=3, label=f'{round(train_metrics[2][-1],4)}  MATTHEW CORRELATION - Training')
# sns.lineplot(val_metrics[2], lw=2,label=f'{round(val_metrics[2][-1],4)}  MATTHEW CORRELATION - Validation')
# sns.lineplot(train_metrics[3], lw=3,label=f'{round(train_metrics[3][-1],4)}  KAPPA SCORE - Training')
# sns.lineplot(val_metrics[3], lw=2, label=f'{round(val_metrics[3][-1],4)}  KAPPA SCORE - Validation')
# plt.xlabel('Epochs', size=16, fontstyle='italic', weight=900)
# plt.ylabel('Accuracy', size=16, fontstyle='italic', weight=900) 
# plt.tight_layout()
# plt.show()

# fig = plt.plot(figsize=(4, 4))
# sns.lineplot(train_loss, lw=3, label = f'{round(train_loss[-1],4)}  Loss - Training')
# sns.lineplot(val_loss, lw=2, label = f'{round(val_loss[-1],4)}  Loss - Validation')
# # plt.legend(labels = [f'{round(train_loss[-1],4)}  Loss - Training', f'{round(val_loss[-1],4)}  Loss - Validation'], fancybox=True)
# plt.xlabel('Epochs', size=16, fontstyle='italic', weight=900)
# plt.ylabel('Loss', size=16, fontstyle='italic', weight=900) 
# plt.tight_layout()
# plt.show()
