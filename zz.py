import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import backend as K



def root_mean_squared_error(pred, test):
    return np.sqrt(((pred - test) ** 2).mean())


def rrmse(pred, test):
    return np.sqrt((((pred - test) / (np.max(test) - np.min(test))) ** 2).mean())


if __name__ == "__main__":
    cnt = 0
    aarray_x = 0
    aarray_y = 0
    parray_x = 0
    parray_y = 0
    ama = ["kdh", "kdk", "khk", "kjd", "kjm", "kkd", "lhj", "lhr", "ljh", "mrc", "oec", "os", "phs",
           "pkr", "sdh", "sts", "wsn", "wmy", "ymc", "ysp", "yym"]
    pro = ["hjj", "hms", "hsw", "jje", "jjy", "jsk", "jty", "kbj", "khj", "kms", "kyj", "lhi", "lsba",
           "lsm", "mhh", "pch", "pcj", "sns", "ssw", "syj", "ych"]

    armse_x = []
    armse_y = []
    armse_z = []
    arrmse_x = []
    arrmse_y = []
    arrmse_z = []
    acc_x = []
    acc_y = []
    acc_z = []

    prmse_x = []
    prmse_y = []
    prmse_z = []
    prrmse_x = []
    prrmse_y = []
    prrmse_z = []
    pcc_x = []
    pcc_y = []
    pcc_z = []

    #데이터 정렬
    cnt = 0
    for k in ama:
        outputpath = "./OUT/AMA/ama_moment/{}.csv".format(k)
        outputdata = pd.read_csv(outputpath, sep=',', header=1, index_col=False)
        outputdata = outputdata.drop("Index", axis=1)
        output = outputdata.values
        outputpath = "./nan/new_{}.csv".format(k)
        outputdata = pd.read_csv(outputpath, sep=',', header=0, index_col=False)
        out = outputdata.values

        arrmse_x.append(rrmse(output[:, 0], out[:, 1]))
        arrmse_y.append(rrmse(output[:, 1], out[:, 2]))
        arrmse_z.append(rrmse(output[:, 2], out[:, 3]))
        armse_x.append(root_mean_squared_error(output[:, 0], out[:, 1]))
        armse_y.append(root_mean_squared_error(output[:, 1], out[:, 2]))
        armse_z.append(root_mean_squared_error(output[:, 2], out[:, 3]))

        new_x = np.concatenate(([output[:, 0]], [out[:, 1]]), axis=0)
        new_y = np.concatenate(([output[:, 1]], [out[:, 2]]), axis=0)
        new_z = np.concatenate(([output[:, 2]], [out[:, 3]]), axis=0)
        df_x = pd.DataFrame(new_x).T
        df_y = pd.DataFrame(new_y).T
        df_z = pd.DataFrame(new_z).T
        acc_x.append(df_x.corr(method="pearson")[0][1])
        acc_y.append(df_y.corr(method="pearson")[0][1])
        acc_z.append(df_z.corr(method="pearson")[0][1])

        if cnt > 0:
            aarray_x = np.vstack([aarray_x, output])
            aarray_y = np.vstack([aarray_y, out])
        else:
            aarray_x = output
            aarray_y = out
            cnt = cnt + 1
    cnt = 0
    for k in pro:
        outputpath = "./OUT/PRO/pro_moment/{}.csv".format(k)
        outputdata = pd.read_csv(outputpath, sep=',', header=1, index_col=False)
        outputdata = outputdata.drop("Index", axis=1)
        output = outputdata.values
        outputpath = "./nan/new_{}.csv".format(k)
        outputdata = pd.read_csv(outputpath, sep=',', header=0, index_col=False)
        out = outputdata.values


        prrmse_x.append(rrmse(output[:, 0], out[:, 1]))
        prrmse_y.append(rrmse(output[:, 1], out[:, 2]))
        prrmse_z.append(rrmse(output[:, 2], out[:, 3]))
        prmse_x.append(root_mean_squared_error(output[:, 0], out[:, 1]))
        prmse_y.append(root_mean_squared_error(output[:, 1], out[:, 2]))
        prmse_z.append(root_mean_squared_error(output[:, 2], out[:, 3]))

        new_x = np.concatenate(([output[:, 0]], [out[:, 1]]), axis=0)
        new_y = np.concatenate(([output[:, 1]], [out[:, 2]]), axis=0)
        new_z = np.concatenate(([output[:, 2]], [out[:, 3]]), axis=0)
        df_x = pd.DataFrame(new_x).T
        df_y = pd.DataFrame(new_y).T
        df_z = pd.DataFrame(new_z).T
        pcc_x.append(df_x.corr(method="pearson")[0][1])
        pcc_y.append(df_y.corr(method="pearson")[0][1])
        pcc_z.append(df_z.corr(method="pearson")[0][1])

        if cnt > 0:
            parray_x = np.vstack([parray_x, output])
            parray_y = np.vstack([parray_y, out])
        else:
            parray_x = output
            parray_y = out
            cnt = cnt + 1
        

    armse_x = np.array(armse_x)
    armse_y = np.array(armse_y)
    armse_z = np.array(armse_z)
    arrmse_x = np.array(arrmse_x)
    arrmse_y = np.array(arrmse_y)
    arrmse_z = np.array(arrmse_z)
    acc_x = np.array(acc_x)
    acc_y = np.array(acc_y)
    acc_z = np.array(acc_z)

    prmse_x = np.array(prmse_x)
    prmse_y = np.array(prmse_y)
    prmse_z = np.array(prmse_z)
    prrmse_x = np.array(prrmse_x)
    prrmse_y = np.array(prrmse_y)
    prrmse_z = np.array(prrmse_z)
    pcc_x = np.array(pcc_x)
    pcc_y = np.array(pcc_y)
    pcc_z = np.array(pcc_z)

    #ama
    ss = np.zeros((len(aarray_x[:,0]),1), )
    ss[:, :] = 0.01

    ayx_x = np.arange(aarray_x[:,0].min(),aarray_x[:,0].max(), 0.01)
    anyx_x = acc_x.mean() * ayx_x

    plt.figure()
    plt.title('AMA Correlation Coefficient X')
    plt.scatter(aarray_x[:,0], aarray_y[:,1], s =ss ,c = 'black')
    plt.plot(ayx_x, ayx_x, 'r', label = '1', linewidth = 0.7)
    plt.plot(ayx_x, anyx_x, 'b--', label = str(round(acc_x.mean(),3)), linewidth = 0.7)
    plt.legend()
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    plt.savefig('nan/AMA_CC_X.png')
    plt.close()

    ayx_y = np.arange(aarray_x[:, 1].min(), aarray_x[:, 1].max(), 0.01)
    anyx_y = acc_y.mean() * ayx_y

    plt.figure()
    plt.title('AMA Correlation Coefficient Y')
    plt.scatter(aarray_x[:, 1], aarray_y[:, 2], s=ss, c='black')
    plt.plot(ayx_y, ayx_y, 'r', label='1', linewidth=0.7)
    plt.plot(ayx_y, anyx_y, 'b--', label=str(round(acc_y.mean(), 3)), linewidth=0.7)
    plt.legend()
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    plt.savefig('nan/AMA_CC_Y.png')
    plt.close()

    ayx_z = np.arange(aarray_x[:, 2].min(), aarray_x[:, 2].max(), 0.01)
    anyx_z = acc_z.mean() * ayx_z

    plt.figure()
    plt.title('AMA Correlation Coefficient Z')
    plt.scatter(aarray_x[:, 2], aarray_y[:, 3], s=ss, c='black')
    plt.plot(ayx_z, ayx_z, 'r', label='1', linewidth=0.7)
    plt.plot(ayx_z, anyx_z, 'b--', label=str(round(acc_z.mean(), 3)), linewidth=0.7)
    plt.legend()
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    plt.savefig('nan/AMA_CC_Z.png')
    plt.close()

    # pro
    ss = np.zeros((len(parray_x[:, 0]), 1), )
    ss[:, :] = 0.01

    pyx_x = np.arange(parray_x[:, 0].min(), parray_x[:, 0].max(), 0.01)
    pnyx_x = pcc_x.mean() * pyx_x

    plt.figure()
    plt.title('PRO Correlation Coefficient X')
    plt.scatter(parray_x[:, 0], parray_y[:, 1], s=ss, c='black')
    plt.plot(pyx_x, pyx_x, 'r', label='1', linewidth=0.7)
    plt.plot(pyx_x, pnyx_x, 'b--', label=str(round(pcc_x.mean(), 3)), linewidth=0.7)
    plt.legend()
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    plt.savefig('nan/PRO_CC_X.png')
    plt.close()

    pyx_y = np.arange(parray_x[:, 1].min(), parray_x[:, 1].max(), 0.01)
    pnyx_y = pcc_y.mean() * pyx_y

    plt.figure()
    plt.title('PRO Correlation Coefficient Y')
    plt.scatter(parray_x[:, 1], parray_y[:, 2], s=ss, c='black')
    plt.plot(pyx_y, pyx_y, 'r', label='1', linewidth=0.7)
    plt.plot(pyx_y, pnyx_y, 'b--', label=str(round(pcc_y.mean(), 3)), linewidth=0.7)
    plt.legend()
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    plt.savefig('nan/PRO_CC_Y.png')
    plt.close()

    pyx_z = np.arange(parray_x[:, 2].min(), parray_x[:, 2].max(), 0.01)
    pnyx_z = pcc_z.mean() * pyx_z

    plt.figure()
    plt.title('PRO Correlation Coefficient Z')
    plt.scatter(parray_x[:, 2], parray_y[:, 3], s=ss, c='black')
    plt.plot(pyx_z, pyx_z, 'r', label='1', linewidth=0.7)
    plt.plot(pyx_z, pnyx_z, 'b--', label=str(round(pcc_z.mean(), 3)), linewidth=0.7)
    plt.legend()
    plt.xlabel('Measured')
    plt.ylabel('Predicted')
    plt.savefig('nan/PRO_CC_Z.png')
    plt.close()

    print('ama')
    print('rmse')
    print(str(armse_x.mean()) + ' ' + str(armse_x.mean() - armse_x.min()) + ' ' + str(armse_x.max() -armse_x.mean()))
    print(str(armse_y.mean()) + ' ' + str(armse_y.mean() - armse_y.min()) + ' ' + str(armse_y.max() - armse_y.mean()))
    print(str(armse_z.mean()) + ' ' + str(armse_z.mean() - armse_z.min()) + ' ' + str(armse_z.max() - armse_z.mean()))
    print('rrmse')
    print(str(arrmse_x.mean()) + ' ' + str(arrmse_x.mean() - arrmse_x.min()) + ' ' + str(arrmse_x.max() - arrmse_x.mean()))
    print(str(arrmse_y.mean()) + ' ' + str(arrmse_y.mean() - arrmse_y.min()) + ' ' + str(arrmse_y.max() - arrmse_y.mean()))
    print(str(arrmse_z.mean()) + ' ' + str(arrmse_z.mean() - arrmse_z.min()) + ' ' + str(arrmse_z.max() - arrmse_z.mean()))
    print('cc')
    print(str(acc_x.mean()) + ' ' + str(acc_x.mean() - acc_x.min()) + ' ' + str(acc_x.max() - acc_x.mean()))
    print(str(acc_y.mean()) + ' ' + str(acc_y.mean() - acc_y.min()) + ' ' + str(acc_y.max() - acc_y.mean()))
    print(str(acc_z.mean()) + ' ' + str(acc_z.mean() - acc_z.min()) + ' ' + str(acc_z.max() - acc_z.mean()))
    print('pro')
    print(str(prmse_x.mean()) + ' ' + str(prmse_x.mean() - prmse_x.min()) + ' ' + str(prmse_x.max() - prmse_x.mean()))
    print(str(prmse_y.mean()) + ' ' + str(prmse_y.mean() - prmse_y.min()) + ' ' + str(prmse_y.max() - prmse_y.mean()))
    print(str(prmse_z.mean()) + ' ' + str(prmse_z.mean() - prmse_z.min()) + ' ' + str(prmse_z.max() - prmse_z.mean()))
    print('pro')
    print(str(prrmse_x.mean()) + ' ' + str(prrmse_x.mean() - prrmse_x.min()) + ' ' + str(prrmse_x.max() - prrmse_x.mean()))
    print(str(prrmse_y.mean()) + ' ' + str(prrmse_y.mean() - prrmse_y.min()) + ' ' + str(prrmse_y.max() - prrmse_y.mean()))
    print(str(prrmse_z.mean()) + ' ' + str(prrmse_z.mean() - prrmse_z.min()) + ' ' + str(prrmse_z.max() - prrmse_z.mean()))
    print('cc')
    print(str(pcc_x.mean()) + ' ' + str(pcc_x.mean() - pcc_x.min()) + ' ' + str(pcc_x.max() - pcc_x.mean()))
    print(str(pcc_y.mean()) + ' ' + str(pcc_y.mean() - pcc_y.min()) + ' ' + str(pcc_y.max() - pcc_y.mean()))
    print(str(pcc_z.mean()) + ' ' + str(pcc_z.mean() - pcc_z.min()) + ' ' + str(pcc_z.max() - pcc_z.mean()))