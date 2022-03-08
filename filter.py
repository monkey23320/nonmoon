import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import pandas as pd
from keras import backend as K



def root_mean_squared_error(pred, test):
    return np.sqrt(((pred - test) ** 2).mean())


def rrmse(pred, test):
    return np.sqrt((((pred - test) / (np.max(test) - np.min(test))) ** 2).mean())


if __name__ == "__main__":
    cnt = 0
    array_forinput = 0
    array_foroutput = 0
    cutoff = 10
    camhertz = 120
    b, a = butter(2, cutoff/camhertz, btype='low')
    ama = ["kdh", "kdk", "khk", "kjd", "kjm", "kkd", "lhj", "lhr", "ljh", "mrc", "oec", "os", "phs",
           "pkr", "sdh", "sts", "wsn", "wmy", "ymc", "ysp", "yym"]
    pro = ["hjj", "hms", "hsw", "jje", "jjy", "jsk", "jty", "kbj", "khj", "kms", "kyj", "lhi", "lsba",
           "lsm", "mhh", "pch", "pcj", "sns", "ssw", "syj", "ych"]
    for k in ama:
        outputpath = "./nan/AMA/{}.csv".format(k)
        outputdata = pd.read_csv(outputpath, sep=',', header=0, index_col=False)
        out = outputdata.values
        new = np.zeros((len(out[:,1]), 3), )
        new[:,0] = filtfilt(b,a,out[:,1])
        new[:,1] = filtfilt(b,a,out[:,2])
        new[:,2] = filtfilt(b,a,out[:,3])
        df = pd.DataFrame(new)
        df.to_csv("nan/new_{}.csv".format(k))
    for k in pro:
        outputpath = "./nan/PRO/{}.csv".format(k)
        outputdata = pd.read_csv(outputpath, sep=',', header=0, index_col=False)
        out = outputdata.values
        new = np.zeros((len(out[:,1]), 3), )
        new[:,0] = filtfilt(b,a,out[:,1])
        new[:,1] = filtfilt(b,a,out[:,2])
        new[:,2] = filtfilt(b,a,out[:,3])
        df = pd.DataFrame(new)
        df.to_csv("nan/new_{}.csv".format(k))
    """
    rmse_x = []
    rmse_y = []
    rmse_z = []
    rrmse_x = []
    rrmse_y = []
    rrmse_z = []
    cc_x = []
    cc_y = []
    cc_z = []
    #데이터 정렬
    for k in pro:
        outputpath = "./OUT/PRO/pro_moment/{}.csv".format(k)
        outputdata = pd.read_csv(outputpath, sep=',', header=1, index_col=False)
        outputdata = outputdata.drop("Index", axis=1)
        output = outputdata.values
        outputpath = "./nan/{}.csv".format(k)
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
        cc_x.append(df_x.corr(method="pearson")[0][1])
        cc_y.append(df_y.corr(method="pearson")[0][1])
        cc_z.append(df_z.corr(method="pearson")[0][1])
    rmse_x = np.array(rmse_x)
    rmse_y = np.array(rmse_y)
    rmse_z = np.array(rmse_z)
    rrmse_x = np.array(rrmse_x)
    rrmse_y = np.array(rrmse_y)
    rrmse_z = np.array(rrmse_z)
    cc_x = np.array(cc_x)
    cc_y = np.array(cc_y)
    cc_z = np.array(cc_z)
    """
