import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential
from sklearn.model_selection import train_test_split


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

    model = Sequential()

    for i in range(2):
        model.add(LSTM(32, batch_input_shape = (1,10,1), stateful=True, return_sequences=True))
        model.add(Dropout(0.3))
    model.add(LSTM(32, stateful=True))
    model.add(Dropout(0.3))
    model.add(Dense(3))
    model.compile(optimizer='adam', loss='mse')

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
    input_train, input_val, output_train, output_val = train_test_split(array_forinput, array_foroutput,
                                                                 test_size=0.1, shuffle=False)

    model.fit(input_train, output_train, epochs=200, batch_size=1, validation_data=(input_val, output_val))






    """input_train, input_val, output_train, output_val = train_test_split(input_train, output_train, shuffle=False,
                                                                        test_size=1 / 3)"""

    """model.add(LSTM(32, input_shape=(None, 1)))
    model.add(Dense(24, activation="relu"))
    model.add(Dense(8, activation="sigmoid"))
    model.add(Dropout(0.3))
    model.add(Dense(3))
    model.compile(optimizer='adam', loss=root_mean_squared_error)"""



    model.save('hello.h5')
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
        input = array.reshape((array.shape[0], array.shape[1], 1))
        output = outputdata.values
        t = np.arange(len(
            output[:, 0]))
        predictions = np.zeros((len(inputdata.index), 3))
        for i in range(len(inputdata.index)):
            xhat = input[i]
            prediction = model.predict(np.array([xhat]), batch_size=1)
            predictions[i] = prediction

        df = pd.DataFrame(predictions)
        df.to_csv("nan/{}.csv".format(k))

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
        input = array.reshape((array.shape[0], array.shape[1], 1))
        output = outputdata.values
        t = np.arange(len(
            output[:, 0]))

        predictions = np.zeros((len(inputdata.index), 3))
        for i in range(len(inputdata.index)):
            xhat = input[i]
            prediction = model.predict(np.array([xhat]), batch_size=1)
            predictions[i] = prediction

        df = pd.DataFrame(predictions)
        df.to_csv("nan/{}.csv".format(k))

    model.summary()
    """loss_mse = model.evaluate(input_test, output_test, batch_size=1)
    print("loss: ", loss_mse)
    out = model.predict(input_test)
    rrmse = rrmse(output_test, out)
    rmse = root_mean_squared_error(output_test, out)
    print("test rrmse: ", rrmse)
    print("test rmse: ", rmse)"""