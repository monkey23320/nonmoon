import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import backend as K

def root_mean_squared_error(y_true, y_pred):
    a=K.sqrt(K.mean(K.square(y_pred - y_true)))
    return float(a)


def rrmse(pred, test):
    return np.sqrt((((pred - test) / (np.max(test) - np.min(test))) ** 2).mean())

if __name__ == "__main__":
    ama = ["kdh", "kdk", "khk", "kjd", "kjm", "kkd", "lhj", "lhr", "ljh", "mrc", "oec", "os", "phs",
           "pkr", "sdh", "sts", "wsn", "wmy", "ymc", "ysp", "yym"]
    pro = ["hjj", "hms", "hsw", "jje", "jjy", "jsk", "jty", "kbj", "khj", "kms", "kyj", "lhi", "lsba",
           "lsm", "mhh", "pch", "pcj", "sns", "ssw", "syj", "ych"]
    for k in ama:
        rmse_x = []
        rmse_y = []
        rmse_z = []
        rrmse_x = []
        rrmse_y = []
        rrmse_z = []
        corr_x = []
        corr_y = []
        corr_z = []
        outputpath = "./OUT/AMA/ama_moment/{}.csv".format(k)
        outputdata = pd.read_csv(outputpath, sep=',', header=1, index_col=False)
        outputdata = outputdata.drop("Index", axis=1)
        output = outputdata.values
        for i in range(1,11):
            outputpath = "./{}/{}.csv".format(i, k)
            outputdata = pd.read_csv(outputpath, sep=',', header=0, index_col=False)
            out = outputdata.values
            rrmse_x.append(rrmse(output[:, 0], out[:, 1]))
            rrmse_y.append(rrmse(output[:, 1], out[:, 2]))
            rrmse_z.append(rrmse(output[:, 2], out[:, 3]))
            rmse_x.append(root_mean_squared_error(output[:, 0], out[:, 1]))
            rmse_y.append(root_mean_squared_error(output[:, 1], out[:, 2]))
            rmse_z.append(root_mean_squared_error(output[:, 2], out[:, 3]))
            new_x = np.concatenate(([output[:, 0]], [out[:, 1]]), axis=0)
            new_y = np.concatenate(([output[:, 1]], [out[:, 2]]), axis=0)
            new_z = np.concatenate(([output[:, 2]], [out[:, 3]]), axis=0)
            df_x = pd.DataFrame(new_x).T
            df_y = pd.DataFrame(new_y).T
            df_z = pd.DataFrame(new_z).T
            corr_x.append(df_x.corr(method="pearson")[0][1])
            corr_y.append(df_y.corr(method="pearson")[0][1])
            corr_z.append(df_z.corr(method="pearson")[0][1])
        rmse_x_mean = np.mean(rmse_x)
        rmse_x_min = np.min(rmse_x)
        rmse_x_max = np.max(rmse_x)
        rmse_y_mean = np.mean(rmse_y)
        rmse_y_min = np.min(rmse_y)
        rmse_y_max = np.max(rmse_y)
        rmse_z_mean = np.mean(rmse_z)
        rmse_z_min = np.min(rmse_z)
        rmse_z_max = np.max(rmse_z)
        rrmse_x_mean = np.mean(rrmse_x)
        rrmse_x_min = np.min(rrmse_x)
        rrmse_x_max = np.max(rrmse_x)
        rrmse_y_mean = np.mean(rrmse_y)
        rrmse_y_min = np.min(rrmse_y)
        rrmse_y_max = np.max(rrmse_y)
        rrmse_z_mean = np.mean(rrmse_z)
        rrmse_z_min = np.min(rrmse_z)
        rrmse_z_max = np.max(rrmse_z)
        corr_x_mean = np.mean(corr_x)
        corr_x_min = np.min(corr_x)
        corr_x_max = np.max(corr_x)
        corr_y_mean = np.mean(corr_y)
        corr_y_min = np.min(corr_y)
        corr_y_max = np.max(corr_y)
        corr_z_mean = np.mean(corr_z)
        corr_z_min = np.min(corr_z)
        corr_z_max = np.max(corr_z)
        print("{} rmse min max rrmse min max corr min max\nX ".format(k) + str(rmse_x_mean) + " " + str(rmse_x_min) + " " + str(
            rmse_x_max) + " " +str(rrmse_x_mean) + " " + str(rrmse_x_min) + " " + str(rrmse_x_max) + " " + str(corr_x_mean) + " " + str(
            corr_x_min) + " " + str(corr_x_max) + "\nY " + str(rmse_y_mean) + " " + str(rmse_y_min) + " " + str(
            rmse_y_max) + " " +str(rrmse_y_mean) + " " + str(rrmse_y_min) + " " + str(rrmse_y_max) + " " + str(corr_y_mean) + " " + str(
            corr_y_min) + " " + str(corr_y_max) + "\nZ " + str(rmse_z_mean) + " " + str(rmse_z_min) + " " + str(
            rmse_z_max) + " " +str(rrmse_z_mean) + " " + str(rrmse_z_min) + " " + str(rrmse_z_max) + " " + str(corr_z_mean) + " " + str(
            corr_z_min) + " " + str(corr_z_max) + "\n")

    for k in pro:
        rmse_x = []
        rmse_y = []
        rmse_z = []
        rrmse_x = []
        rrmse_y = []
        rrmse_z = []
        corr_x = []
        corr_y = []
        corr_z = []
        outputpath = "./OUT/PRO/pro_moment/{}.csv".format(k)
        outputdata = pd.read_csv(outputpath, sep=',', header=1, index_col=False)
        outputdata = outputdata.drop("Index", axis=1)
        output = outputdata.values
        for i in range(1,11):
            outputpath = "./{}/{}.csv".format(i, k)
            outputdata = pd.read_csv(outputpath, sep=',', header=0, index_col=False)
            out = outputdata.values
            rrmse_x.append(rrmse(output[:, 0], out[:, 1]))
            rrmse_y.append(rrmse(output[:, 1], out[:, 2]))
            rrmse_z.append(rrmse(output[:, 2], out[:, 3]))
            rmse_x.append(root_mean_squared_error(output[:, 0], out[:, 1]))
            rmse_y.append(root_mean_squared_error(output[:, 1], out[:, 2]))
            rmse_z.append(root_mean_squared_error(output[:, 2], out[:, 3]))
            new_x = np.concatenate(([output[:, 0]],[ out[:, 1]]))
            new_y = np.concatenate(([output[:, 1]], [out[:, 2]]))
            new_z = np.concatenate(([output[:, 2]], [out[:, 3]]))
            df_x = pd.DataFrame(new_x).T
            df_y = pd.DataFrame(new_y).T
            df_z = pd.DataFrame(new_z).T
            x = df_x.corr(method="pearson")
            y = df_y.corr(method="pearson")
            z = df_z.corr(method="pearson")
            corr_x.append(x[0][1])
            corr_y.append(y[0][1])
            corr_z.append(z[0][1])
        rmse_x_mean = np.mean(rmse_x)
        rmse_x_min = np.min(rmse_x)
        rmse_x_max = np.max(rmse_x)
        rmse_y_mean = np.mean(rmse_y)
        rmse_y_min = np.min(rmse_y)
        rmse_y_max = np.max(rmse_y)
        rmse_z_mean = np.mean(rmse_z)
        rmse_z_min = np.min(rmse_z)
        rmse_z_max = np.max(rmse_z)
        rrmse_x_mean = np.mean(rrmse_x)
        rrmse_x_min = np.min(rrmse_x)
        rrmse_x_max = np.max(rrmse_x)
        rrmse_y_mean = np.mean(rrmse_y)
        rrmse_y_min = np.min(rrmse_y)
        rrmse_y_max = np.max(rrmse_y)
        rrmse_z_mean = np.mean(rrmse_z)
        rrmse_z_min = np.min(rrmse_z)
        rrmse_z_max = np.max(rrmse_z)
        corr_x_mean = np.mean(corr_x)
        corr_x_min = np.min(corr_x)
        corr_x_max = np.max(corr_x)
        corr_y_mean = np.mean(corr_y)
        corr_y_min = np.min(corr_y)
        corr_y_max = np.max(corr_y)
        corr_z_mean = np.mean(corr_z)
        corr_z_min = np.min(corr_z)
        corr_z_max = np.max(corr_z)
        print("{} rmse min max rrmse min max corr min max\nX ".format(k) + str(rmse_x_mean) + " " + str(rmse_x_min) + " " + str(
            rmse_x_max) + " " +str(rrmse_x_mean) + " " + str(rrmse_x_min) + " " + str(rrmse_x_max) + " " + str(corr_x_mean) + " " + str(
            corr_x_min) + " " + str(corr_x_max) + "\nY " + str(rmse_y_mean) + " " + str(rmse_y_min) + " " + str(
            rmse_y_max) + " " +str(rrmse_y_mean) + " " + str(rrmse_y_min) + " " + str(rrmse_y_max) + " " + str(corr_y_mean) + " " + str(
            corr_y_min) + " " + str(corr_y_max) + "\nZ " + str(rmse_z_mean) + " " + str(rmse_z_min) + " " + str(
            rmse_z_max) + " " +str(rrmse_z_mean) + " " + str(rrmse_z_min) + " " + str(rrmse_z_max) + " " + str(corr_z_mean) + " " + str(
            corr_z_min) + " " + str(corr_z_max) + "\n")