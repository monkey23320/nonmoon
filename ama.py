import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def root_mean_squared_error(pred, test):
    return np.sqrt(((pred - test) ** 2).mean())

if __name__ == "__main__":
    ama = ["kdh", "kdk", "khk", "kjd", "kjm", "kkd", "lhj", "lhr", "ljh", "mrc", "oec", "os", "phs",
           "pkr", "sdh", "sts", "wsn", "wmy", "ymc", "ysp", "yym"]
    pro = ["hjj", "hms", "hsw", "jje", "jjy", "jsk", "jty", "kbj", "khj", "kms", "kyj", "lhi", "lsba",
           "lsm", "mhh", "pch", "pcj", "sns", "ssw", "syj", "ych"]
    cnt = 0
    for k in ama:

        outputpath = "./OUT/AMA/ama_moment/{}.csv".format(k)
        outputdata = pd.read_csv(outputpath, sep=',', header=1, index_col=False)
        outputdata = outputdata.drop("Index", axis=1)
        output = outputdata.values
        outputpath = "./nan/AMA/new_{}.csv".format( k)
        outputdata = pd.read_csv(outputpath, sep=',', header=0, index_col=False)
        out = outputdata.values

        t = np.arange(len(output[:,0]))

        plt.figure()
        plt.title('{} L5S1 X Torque'.format(k))
        plt.plot(t, output[:, 0], 'r', label='measured')
        plt.plot(t, out[:, 1], 'b--',label='predicted')
        plt.legend()
        plt.xlabel('frame')
        plt.ylabel('x_torque(N·m/Kg)')
        plt.savefig('nan/AMA/{}_X.png'.format( k))
        plt.close()

        plt.figure()
        plt.title('{} L5S1 Y Torque'.format(k))
        plt.plot(t, output[:, 1], 'r', label='measured')
        plt.plot(t, out[:, 2], 'b--',  label='predicted')
        plt.legend()
        plt.xlabel('frame')
        plt.ylabel('y_torque(N·m/Kg)')
        plt.savefig('nan/AMA/{}_Y.png'.format( k))
        plt.close()

        plt.figure()
        plt.title('{} L5S1 Z Torque'.format(k))
        plt.plot(t, output[:, 2], 'r', label='measured')
        plt.plot(t, out[:, 3], 'b--',  label='predicted')
        plt.legend()
        plt.xlabel('frame')
        plt.ylabel('z_torque(N·m/Kg)')
        plt.savefig('nan/AMA/{}_Z.png'.format( k))
        plt.close()

    for k in pro:


        outputpath = "./OUT/PRO/pro_moment/{}.csv".format(k)
        outputdata = pd.read_csv(outputpath, sep=',', header=1, index_col=False)
        outputdata = outputdata.drop("Index", axis=1)
        output = outputdata.values
        outputpath = "./nan/PRO/new_{}.csv".format( k)
        outputdata = pd.read_csv(outputpath, sep=',', header=0, index_col=False)
        out = outputdata.values

        t = np.arange(len(output[:, 0]))
        plt.figure()
        plt.title('{} L5S1 X Torque'.format(k))
        plt.plot(t, output[:, 0], 'r', label='measured')
        plt.plot(t, out[:, 1], 'b--', label='predicted')
        plt.legend()
        plt.xlabel('frame')
        plt.ylabel('x_torque(N·m/Kg)')
        plt.savefig('nan/PRO/{}_X.png'.format(k))
        plt.close()

        plt.figure()
        plt.title('{} L5S1 Y Torque'.format(k))
        plt.plot(t, output[:, 1], 'r', label='measured')
        plt.plot(t, out[:, 2], 'b--', label='predicted')
        plt.legend()
        plt.xlabel('frame')
        plt.ylabel('y_torque(N·m/Kg)')
        plt.savefig('nan/PRO/{}_Y.png'.format(k))
        plt.close()

        plt.figure()
        plt.title('{} L5S1 Z Torque'.format(k))
        plt.plot(t, output[:, 2], 'r', label='measured')
        plt.plot(t, out[:, 3], 'b--', label='predicted')
        plt.legend()
        plt.xlabel('frame')
        plt.ylabel('z_torque(N·m/Kg)')
        plt.savefig('nan/PRO/{}_Z.png'.format(k))
        plt.close()