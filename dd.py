import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


def rrmse(pred, test):
    return np.sqrt((((pred - test) / (np.max(test) - np.min(test))) ** 2).mean())


if __name__ == "__main__":
    cnt = 0
    array_forinput = 0
    array_foroutput = 0
    ama = ["kdh", "kdk", "khk", "kjd", "kjm", "kkd", "lhj", "lhr", "ljh", "mrc", "oec", "os", "phs",
           "pkr", "sdh", "sts", "wsn", "wmy", "ymc", "ysp", "yym"]
    pro = ["hjj", "hms", "hsw", "jje", "jjy", "jsk", "jty", "kbj", "khj", "kms", "kyj", "lhi", "lsba",
           "lsm", "mhh", "pch", "pcj", "sns", "ssw", "syj", "ych"]
    #데이터 정렬
    for k in ama:
        outputpath = "./OUT/AMA/ama_moment/{}.csv".format(k)
        outputdata = pd.read_csv(outputpath, sep=',', header=1, index_col=False)
        outputdata = outputdata.drop("Index", axis=1)
        array_width = 10
        inputpath = "./IN/AMA/ama_txt/dynamic_{}1.txt".format(k)
        inputdata = pd.read_csv(inputpath, sep='\t', header=None)
        array = np.zeros((len(inputdata.index), array_width), dtype=np.float)
        for i in range(0, len(inputdata.index)):
            array[i, 0] = inputdata.iloc[i][111]
            array[i, 1] = inputdata.iloc[i][112]
            array[i, 2] = inputdata.iloc[i][113]
            array[i, 3] = inputdata.iloc[i][114]
            array[i, 4] = inputdata.iloc[i][115]
            array[i, 5] = inputdata.iloc[i][116]
            array[i, 6] = inputdata.iloc[i][117]
            array[i, 7] = inputdata.iloc[i][118]
            array[i, 8] = inputdata.iloc[i][120]
            array[i, 9] = inputdata.iloc[i][121]
        output = outputdata.values
        if cnt != 0:
            array_forinput = np.vstack([array_forinput, array])
            array_foroutput = np.vstack([array_foroutput, output])
            continue
        array_forinput = array
        array_foroutput = outputdata.values
        cnt += 1

    for k in pro:
        outputpath = "./OUT/PRO/pro_moment/{}.csv".format(k)
        outputdata = pd.read_csv(outputpath, sep=',', header=1, index_col=False)
        outputdata = outputdata.drop("Index", axis=1)
        array_width = 10
        inputpath = "./IN/PRO/pro_txt/dynamic_{}1.txt".format(k)
        inputdata = pd.read_csv(inputpath, sep='\t', header=None)
        array = np.zeros((len(inputdata.index), array_width), dtype=np.float)
        for i in range(0, len(inputdata.index)):
            array[i, 0] = inputdata.iloc[i][111]
            array[i, 1] = inputdata.iloc[i][112]
            array[i, 2] = inputdata.iloc[i][113]
            array[i, 3] = inputdata.iloc[i][114]
            array[i, 4] = inputdata.iloc[i][115]
            array[i, 5] = inputdata.iloc[i][116]
            array[i, 6] = inputdata.iloc[i][117]
            array[i, 7] = inputdata.iloc[i][118]
            array[i, 8] = inputdata.iloc[i][120]
            array[i, 9] = inputdata.iloc[i][121]
        output = outputdata.values
        if cnt != 0:
            array_forinput = np.vstack([array_forinput, array])
            array_foroutput = np.vstack([array_foroutput, output])
            continue
        array_forinput = array
        array_foroutput = outputdata.values
        cnt += 1
    array_forinput = array_forinput.reshape((array_forinput.shape[0], array_forinput.shape[1], 1))
    #정렬 끝
    base = 1
    skf = KFold(n_splits=10, shuffle=False)
    cnt = 1
    for train, validation in skf.split(array_forinput, array_foroutput):
        input_train, in_test, output_train, out_test = train_test_split(array_forinput[train], array_foroutput[train], shuffle=False, test_size=1 / 9)
        model = Sequential()
        model.add(LSTM(32, input_shape=(None, 1)))
        model.add(Dropout(0.3))
        model.add(Dense(3))
        model.compile(optimizer='adam', loss=root_mean_squared_error)
        model.fit(input_train, output_train, epochs=100, batch_size=1, validation_data=(array_forinput[validation], array_foroutput[validation]))

        model.summary()
        loss_mse = model.evaluate(in_test, out_test, batch_size=1)
        print(base)
        print("loss: ", loss_mse)
        out = model.predict(in_test)
        rrmse = rrmse(out_test, out)
        rmse = root_mean_squared_error(out_test, out)
        print("test rrmse: ", rrmse)
        print("test rmse: ", rmse)
    """input_train, input_test, output_train, output_test = train_test_split(array_forinput, array_foroutput,
                                                                          test_size=0.2, shuffle=False)
    input_train, input_val, output_train, output_val = train_test_split(input_train, output_train, shuffle=False,
                                                                        test_size=1 / 3)"""
    """model = Sequential()
    model.add(LSTM(32, input_shape=(None, 1)))
    model.add(Dropout(0.3))
    model.add(Dense(3))
    model.compile(optimizer='adam', loss=root_mean_squared_error)
    model.fit(input_train, output_train, epochs=100, batch_size=1, validation_data=(input_val, output_val))"""

    #사진 뽑아내기

    for k in ama:
        outputpath = "./OUT/AMA/ama_moment/{}.csv".format(k)
        outputdata = pd.read_csv(outputpath, sep=',', header=1, index_col=False)
        outputdata = outputdata.drop("Index", axis=1)
        array_width = 10
        inputpath = "./IN/AMA/ama_txt/dynamic_{}1.txt".format(k)
        inputdata = pd.read_csv(inputpath, sep='\t', header=None)
        array = np.zeros((len(inputdata.index), array_width), dtype=np.float)
        for i in range(0, len(inputdata.index)):
            array[i, 0] = inputdata.iloc[i][111]
            array[i, 1] = inputdata.iloc[i][112]
            array[i, 2] = inputdata.iloc[i][113]
            array[i, 3] = inputdata.iloc[i][114]
            array[i, 4] = inputdata.iloc[i][115]
            array[i, 5] = inputdata.iloc[i][116]
            array[i, 6] = inputdata.iloc[i][117]
            array[i, 7] = inputdata.iloc[i][118]
            array[i, 8] = inputdata.iloc[i][120]
            array[i, 9] = inputdata.iloc[i][121]
        array_forinput = array.reshape((array.shape[0], array.shape[1], 1))
        array_foroutput = outputdata.values
        out = model.predict(array_forinput)
        t = np.arange(len(array_foroutput[:, 0]))
        plt.figure(cnt)
        cnt += 1
        plt.title('{} L5S1 X moment'.format(k))
        plt.plot(t, array_foroutput[:, 0], 'r', label='measured')
        plt.plot(t, out[:, 0], 'b', label='predicted')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('x_moment')
        plt.savefig('{}/{}_X.png'.format(base, k))
        plt.figure(cnt)
        cnt += 1
        plt.title('{} L5S1 Y moment'.format(k))
        plt.plot(t, array_foroutput[:, 1], 'r', label='measured')
        plt.plot(t, out[:, 1], 'b', label='predicted')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('y_moment')
        plt.savefig('{}/{}_Y.png'.format(base, k))
        plt.figure(cnt)
        cnt += 1
        plt.title('{} L5S1 Z moment'.format(k))
        plt.plot(t, array_foroutput[:, 2], 'r', label='measured')
        plt.plot(t, out[:, 2], 'b', label='predicted')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('z_moment')
        plt.savefig('{}/{}_Z.png'.format(base, k))

    for k in pro:
        outputpath = "./OUT/PRO/pro_moment/{}.csv".format(k)
        outputdata = pd.read_csv(outputpath, sep=',', header=1, index_col=False)
        outputdata = outputdata.drop("Index", axis=1)
        array_width = 10
        inputpath = "./IN/PRO/pro_txt/dynamic_{}1.txt".format(k)
        inputdata = pd.read_csv(inputpath, sep='\t', header=None)
        array = np.zeros((len(inputdata.index), array_width), dtype=np.float)
        for i in range(0, len(inputdata.index)):
            array[i, 0] = inputdata.iloc[i][111]
            array[i, 1] = inputdata.iloc[i][112]
            array[i, 2] = inputdata.iloc[i][113]
            array[i, 3] = inputdata.iloc[i][114]
            array[i, 4] = inputdata.iloc[i][115]
            array[i, 5] = inputdata.iloc[i][116]
            array[i, 6] = inputdata.iloc[i][117]
            array[i, 7] = inputdata.iloc[i][118]
            array[i, 8] = inputdata.iloc[i][120]
            array[i, 9] = inputdata.iloc[i][121]
        array_forinput = array.reshape((array.shape[0], array.shape[1], 1))
        array_foroutput = outputdata.values
        out = model.predict(array_forinput)
        t = np.arange(len(array_foroutput[:, 0]))
        plt.figure(cnt)
        cnt += 1
        plt.title('{} L5S1 X moment'.format(k))
        plt.plot(t, array_foroutput[:, 0], 'r', label='measured')
        plt.plot(t, out[:, 0], 'b', label='predicted')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('x_moment')
        plt.savefig('{}/{}_X.png'.format(base, k))
        plt.figure(cnt)
        cnt += 1
        plt.title('{} L5S1 Y moment'.format(k))
        plt.plot(t, array_foroutput[:, 1], 'r', label='measured')
        plt.plot(t, out[:, 1], 'b', label='predicted')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('y_moment')
        plt.savefig('{}/{}_Y.png'.format(k))
        plt.figure(cnt)
        cnt += 1
        plt.title('{} L5S1 Z moment'.format(k))
        plt.plot(t, array_foroutput[:, 2], 'r', label='measured')
        plt.plot(t, out[:, 2], 'b', label='predicted')
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('z_moment')
        plt.savefig('{}/{}_Z.png'.format(k))
    #사진 뽑기 끝끝
