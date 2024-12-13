import numpy as np

# Example real and predicted values
real_values = [[0.3700, 59.8800, 45.4500], [0.3200, 59.5200, 43.4400],[0.3900, 59.4200, 43.2800],
               [0.4000, 59.3100, 42.6000], [0.3600, 56.8700, 43.9700], [0.3700, 61.8000, 44.7700],
               [0.3400, 57.8200, 43.6300], [0.2900, 57.5300, 44.2800], [0.3600, 59.3300, 46.5800],
               [0.3300, 55.7600, 44.0800], [0.3600, 58.2000, 43.5300]]
unet_predictions = [[0.4479, 64.5472, 48.7350], [0.3876, 61.7045, 46.7352],[0.4246, 63.9901, 48.0945],
               [0.4378, 59.5307, 44.5303], [0.3516, 61.4720, 46.4149], [0.4443, 64.2553, 48.4615],
               [0.3696, 62.4110, 47.3255], [0.4598, 62.3857, 47.2127], [0.3945, 63.9301, 48.2832],
               [0.2728, 60.0560, 45.7268], [0.3209, 61.6495, 46.8414]]
fcn_predictions = [[0.3949, 60.3605, 45.2794], [0.3583, 58.9571, 44.1526],[0.3054, 59.7758, 44.6681],
               [0.3133, 57.7527, 42.9015], [0.2110, 57.4620, 43.0780], [0.4490, 60.7595, 45.2866],
               [0.2664, 58.8440, 44.1922], [0.3108, 58.1743, 44.2006], [0.2259, 59.3228, 44.6234],
               [0.2882, 56.4656, 43.3312], [0.2100, 58.7781, 44.3173]]

# Calculate MAE
unet_mae = np.mean(np.abs(np.array(unet_predictions) - np.array(real_values)))
fcn_mae = np.mean(np.abs(np.array(fcn_predictions) - np.array(real_values)))

print(f"U-Net MAE: {unet_mae}")
print(f"FCN MAE: {fcn_mae}")

unet_rmse = np.sqrt(np.mean((np.array(unet_predictions) - np.array(real_values)) ** 2))
fcn_rmse = np.sqrt(np.mean((np.array(fcn_predictions) - np.array(real_values)) ** 2))

print(f"U-Net RMSE: {unet_rmse}")
print(f"FCN RMSE: {fcn_rmse}")


unet_mape = np.mean(np.abs((np.array(unet_predictions) - np.array(real_values)) / np.array(real_values))) * 100
fcn_mape = np.mean(np.abs((np.array(fcn_predictions) - np.array(real_values)) / np.array(real_values))) * 100

print(f"U-Net MAPE: {unet_mape}%")
print(f"FCN MAPE: {fcn_mape}%")


from scipy.stats import ttest_rel

# Paired t-test for Thickness
t_stat, p_value = ttest_rel(unet_predictions, real_values)
print(f"U-Net T-test P-value: {p_value}")

t_stat, p_value = ttest_rel(fcn_predictions, real_values)
print(f"FCN T-test P-value: {p_value}")