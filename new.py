
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline, BSpline
from keras.models import load_model
from keras import backend as K

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

if __name__ == "__main__":
    ama = ["kdh", "kdk", "khk", "kjd", "kjm", "kkd", "lhj", "lhr", "ljh", "mrc", "oec", "os", "phs",
           "pkr", "sdh", "sts", "wsn", "wmy", "ymc", "ysp", "yym"]
    pro = ["hjj", "hms", "hsw", "jje", "jjy", "jsk", "jty", "kbj", "khj", "kms", "kyj", "lhi", "lsba",
           "lsm", "mhh", "pch", "pcj", "sns", "ssw", "syj", "ych"]
    cnt = 0
    model = load_model("hello.h5")
    for k in ama:
        outputpath = "./OUT/AMA/ama_moment/{}.csv".format(k)
        outputdata = pd.read_csv(outputpath, sep=',', header=1, index_col=False)
        outputdata = outputdata.drop("Index", axis=1)
        output = outputdata.values
        outputpath = "./ss/9/{}.csv".format( k)
        outputdata = pd.read_csv(outputpath, sep=',', header=0, index_col=False)
        out = outputdata.values
        t = np.arange(len(output[:,0]))
        new_t = np.linspace(t.min(), t.max(), 1000)

        plt.figure(cnt)
        cnt += 1
        plt.title('{} L5S1 X moment'.format(k))
        plt.plot(t, output[:, 0], 'r', label='measured')
        spl_x = make_interp_spline(t, out[:,1], k=3)
        x_s = spl_x(new_t)
        plt.plot(new_t, x_s, 'b', label='predicted')

        #plt.plot(t, out[:, 1], 'b', label='predicted')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('x_moment')
        plt.savefig('nan/{}_X.png'.format( k))
        plt.figure(cnt)
        cnt += 1
        plt.title('{} L5S1 Y moment'.format(k))
        plt.plot(t, output[:, 1], 'r', label='measured')
        spl_y = make_interp_spline(t, out[:, 2], k=3)
        y_s = spl_y(new_t)
        plt.plot(new_t, y_s, 'b', label='predicted')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('y_moment')
        plt.savefig('nan/{}_Y.png'.format( k))
        plt.figure(cnt)
        cnt += 1
        plt.title('{} L5S1 Z moment'.format(k))
        plt.plot(t, output[:, 2], 'r', label='measured')
        spl_z = make_interp_spline(t, out[:, 3], k=3)
        z_s = spl_z(new_t)
        plt.plot(new_t, z_s, 'b', label='predicted')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('z_moment')
        plt.savefig('nan/{}_Z.png'.format( k))

    for k in pro:

        outputpath = "./OUT/PRO/pro_moment/{}.csv".format(k)
        outputdata = pd.read_csv(outputpath, sep=',', header=1, index_col=False)
        outputdata = outputdata.drop("Index", axis=1)
        output = outputdata.values
        outputpath = "./nan/{}.csv".format( k)
        outputdata = pd.read_csv(outputpath, sep=',', header=0, index_col=False)
        out = outputdata.values
        t = np.arange(len(output[:, 0]))
        new_t = np.linspace(t.min(), t.max(), 1000)

        plt.figure(cnt)
        cnt += 1
        plt.title('{} L5S1 X moment'.format(k))
        plt.plot(t, output[:, 0], 'r', label='measured')
        spl_x = make_interp_spline(t, out[:, 1], k=3)
        x_s = spl_x(new_t)
        plt.plot(new_t, x_s, 'b', label='predicted')

        # plt.plot(t, out[:, 1], 'b', label='predicted')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('x_moment')
        plt.savefig('nan/{}_X.png'.format(k))
        plt.figure(cnt)
        cnt += 1
        plt.title('{} L5S1 Y moment'.format(k))
        plt.plot(t, output[:, 1], 'r', label='measured')
        spl_y = make_interp_spline(t, out[:, 2], k=3)
        y_s = spl_y(new_t)
        plt.plot(new_t, y_s, 'b', label='predicted')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('y_moment')
        plt.savefig('nan/{}_Y.png'.format(k))
        plt.figure(cnt)
        cnt += 1
        plt.title('{} L5S1 Z moment'.format(k))
        plt.plot(t, output[:, 2], 'r', label='measured')
        spl_z = make_interp_spline(t, out[:, 3], k=3)
        z_s = spl_z(new_t)
        plt.plot(new_t, z_s, 'b', label='predicted')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('z_moment')
        plt.savefig('nan/{}_Z.png'.format(k))