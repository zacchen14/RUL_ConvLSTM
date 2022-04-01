import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def create_train_rul_label(dataframe: pd.DataFrame, mode='train') -> pd.DataFrame:
    for id_train in dataframe.index.unique():
        dataframe.loc[id_train, 'RUL'] = -1 * dataframe.loc[id_train]['time'].apply(
            lambda x: x - dataframe.loc[id_train]['time'].max()
            if (x - dataframe.loc[id_train]['time'].max()) > -125 else -125)
    return dataframe


def create_test_rul_label(dataframe: pd.DataFrame, rulframe: pd.DataFrame, mode='train') -> pd.DataFrame:
    for id_test in dataframe.index.unique():
        dataframe.loc[id_test, 'RUL'] = -1 * dataframe.loc[id_test]['time'].apply(
            lambda x: x - dataframe.loc[id_test]['time'].max() - rulframe.loc[id_test - 1].values[0]
            if (x - dataframe.loc[id_test]['time'].max() - rulframe.loc[id_test - 1].values[
                0]) > -125 else -125)
    return dataframe


def slice_dataset(dataframe: pd.DataFrame, mode="train"):
    print(mode)
    engines = dataframe.index.unique().values
    engine_slices = dict()
    for i, engine_num in enumerate(engines):
        row_name = dataframe.loc[engine_num].iloc[-1].name
        row_sl = dataframe.index.get_loc(row_name)
        engine_slices[engine_num] = row_sl

    x = np.random.randn(0, args.T, 1, args.tw, 16)
    y = np.random.randn(0)
    x_case = np.random.randn(0, args.T, 1, args.tw, 16) if mode == 'test' else None
    y_case = np.random.randn(0) if mode == 'test' else None
    for _, engine_num in enumerate(engines):
        dataframe_engine = dataframe[engine_slices[engine_num]]
        for i in range(args.tw + args.T, dataframe_engine.shape[0]):
            x_sample = np.random.randn(0, args.tw, 16)
            for j in range(args.T, 0, -1):
                x_new_sample = \
                    np.expand_dims(np.array(dataframe_engine.iloc[i - j - args.tw:i - j, 0:16]), axis=0)
                x_sample = np.concatenate((x_sample, x_new_sample), axis=0)
            y_sample = np.array(dataframe_engine.iloc[i, 16])

            x_sample = np.expand_dims(x_sample, axis=[0, 2])
            y_sample = np.expand_dims(y_sample, axis=-1)
            x = np.concatenate((x, x_sample), axis=0)
            y = np.concatenate((y, y_sample), axis=0)
        if mode == 'test':
            x_case = np.concatenate((x_case, x_sample), axis=0)
            y_case = np.concatenate((y_case, y_sample), axis=0)
        print('The engine', engine_num, 'is processing,', 'Current shape is', x.shape, y.shape)

    return x, y, x_case, y_case


def pre_process(input: str, output: str):
    """
    Read the .txt file and add RUL to each engine based on time column,
    notice that RUL is negative quantity here to make 0 as the end of life for all engines.
    Specifically, as for test dataset, RUL should added by RUL label in RUL_FD00x.txt.

    Args:
        output: the file path of output data
    """

    path = os.path.join(os.getcwd(), 'data', 'CMAPSSData')
    col_name = ['engine', 'time', 'op_cond_1', 'op_cond_2', 'op_cond_3'] + ['sn_{}'.format(s + 1) for s in
                                                                            range(21)]
    dataset = 'FD004'
    print(dataset)
    # index_col = engine
    train_dataframe = pd.read_csv(os.path.join(path, 'train_FD004.txt'), header=None, names=col_name,
                                  delim_whitespace=True, index_col=0)
    test_dataframe = pd.read_csv(os.path.join(path, 'test_FD004.txt'), header=None, names=col_name,
                                 delim_whitespace=True, index_col=0)
    rul_dataframe = pd.read_csv(os.path.join(path, 'RUL_FD004.txt'), header=None, names=['RUL'],
                                delim_whitespace=True)

    # create RUL label
    # train_dataframe = create_train_rul_label(train_dataframe, mode='train')
    # test_dataframe = create_test_rul_label('test', test_dataframe, rul_dataframe)
    # Equal to above 2 lines.

    for id_train in train_dataframe.index.unique():
        train_dataframe.loc[id_train, 'RUL'] = -1 * train_dataframe.loc[id_train]['time'].apply(
            lambda x: x - train_dataframe.loc[id_train]['time'].max()
            if (x - train_dataframe.loc[id_train]['time'].max()) > -125 else -125)

    for id_test in test_dataframe.index.unique():
        test_dataframe.loc[id_test, 'RUL'] = -1 * test_dataframe.loc[id_test]['time'].apply(
            lambda x: x - test_dataframe.loc[id_test]['time'].max() - rul_dataframe.loc[id_test - 1].values[0]
            if (x - test_dataframe.loc[id_test]['time'].max() - rul_dataframe.loc[id_test - 1].values[
                0]) > -125 else -125)

    # drop stable features
    train_dataframe.drop(['time', 'op_cond_3', 'sn_1', 'sn_5', 'sn_6', 'sn_10', 'sn_16', 'sn_18', 'sn_19'],
                         axis=1, inplace=True)
    test_dataframe.drop(['time', 'op_cond_3', 'sn_1', 'sn_5', 'sn_6', 'sn_10', 'sn_16', 'sn_18', 'sn_19'],
                        axis=1, inplace=True)

    # Data Normalization
    train_dataframe.iloc[:, :-1] = (train_dataframe.iloc[:, :-1] - train_dataframe.iloc[:, :-1].mean()) / \
                                   train_dataframe.iloc[:, :-1].std()
    test_dataframe.iloc[:, :-1] = (test_dataframe.iloc[:, :-1] - test_dataframe.iloc[:, :-1].mean()) / \
                                  test_dataframe.iloc[:, :-1].std()

    # ====  Preprocess train dataset  ===
    # train_engine_slices： dictionary.
    ''' 
    key: row slice, value: a slice that gives numpy index for the data that pertains to an engine
    For instance, train_engine_slices = {..., '4': Slice(400, 480, ), '5': Slice(481, 560, ), ...}
    - Shape: 
        (n, input_channel=1, height=args.twlen, width=feature_num),  
    where n is the number of samples.
    '''

    # train_x, train_y, _, _ = slice_dataset(train_dataframe)

    # np.save(file=output + "/train_x_FD004.npy", arr=train_x)
    # np.save(file=output + "/train_y_FD004.npy", arr=train_y)

    # ====  Preprocess test dataset  ===
    # train_engine_slices： dictionary.
    # key: row slice, value: a slice that gives numpy index for the data that pertains to an engine
    # For instance, train_engine_slices = {..., '4': Slice(400, 480, ), '5': Slice(481, 560, ), ...}
    # -Shape: (n, input_channel=1, height=args.twlen, width=feature_num),
    # where n is the number of samples.
    test_x, test_y, case_x, case_y = slice_dataset(test_dataframe, mode='test')
    print('s')
    np.save(file=output + "/test_x_FD004.npy", arr=test_x)
    np.save(file=output + "/test_y_FD004.npy", arr=test_y)
    np.save(file=output + "/case_x_FD004.npy", arr=case_x)
    np.save(file=output + "/case_y_FD004.npy", arr=case_y)

    '''
    train_engines = train_dataframe.index.unique().values
    train_engine_slices = dict()
    for i, engine_num in enumerate(train_engines):
        row_name = train_dataframe.loc[engine_num].iloc[-1].name
        row_sl = train_dataframe.index.get_loc(row_name)
        train_engine_slices[engine_num] = row_sl

    x = np.random.randn(0, args.T, 1, args.tw, 16)
    y = np.random.randn(0)
    for _, engine_num in enumerate(train_engines):
        train_dataframe_engine = train_dataframe[train_engine_slices[engine_num]]
        for i in range(args.tw + args.T, train_dataframe_engine.shape[0]):
            x_sample = np.random.randn(0, args.tw, 16)
            for j in range(args.T, 0, -1):
                x_new_sample = \
                    np.expand_dims(np.array(train_dataframe_engine.iloc[i - j - args.tw:i - j, 0:16]), axis=0)
                x_sample = np.concatenate((x_sample, x_new_sample), axis=0)

            y_sample = np.array(train_dataframe_engine.iloc[i, 16])
            x_sample = np.expand_dims(x_sample, axis=[0, 2])
            y_sample = np.expand_dims(y_sample, axis=-1)
            x = np.concatenate((x, x_sample), axis=0)
            y = np.concatenate((y, y_sample), axis=0)
        print('The engine', engine_num, 'is processing,', 'Current shape is', x.shape, y.shape)

    np.save(file=output + "/train_x_FD001.npy", arr=x)
    np.save(file=output + "/train_y_FD001.npy", arr=y)
    print('The .npy file has been saved to', )


    # Preprocess test dataset
    test_engines = test_dataframe.index.unique().values
    test_engine_slices = dict()
    for i, engine_num in enumerate(test_engines):
        row_name = test_dataframe.loc[engine_num].iloc[-1].name
        row_sl = test_dataframe.index.get_loc(row_name)  # row slice to get numpy index
        test_engine_slices[engine_num] = row_sl

    x = np.random.randn(0, args.T, 1, args.tw, 16)
    y = np.random.randn(0)

    x_for_valid = np.random.randn(0, args.T, 1, args.tw, 16)
    for _, engine_num in enumerate(test_engines):
        test_dataframe_engine = test_dataframe[test_engine_slices[engine_num]]
        for i in range(args.tw + args.T, test_dataframe_engine.shape[0]):
            x_sample = np.random.randn(0, args.tw, 16)
            for j in range(args.T, 0, -1):
                x_new_sample = np.expand_dims(np.array(test_dataframe_engine.iloc[i - j - args.tw:i - j, 0:16]),
                                              axis=0)
                x_sample = np.concatenate((x_sample, x_new_sample), axis=0)
            y_sample = np.array(test_dataframe_engine.iloc[i, 16])

            x_sample = np.expand_dims(x_sample, axis=[0, 2])
            y_sample = np.expand_dims(y_sample, axis=-1)
            x = np.concatenate((x, x_sample), axis=0)
            y = np.concatenate((y, y_sample), axis=0)
        x_for_valid = np.concatenate((x_for_valid, x_sample), axis=0)
        print('Engine', engine_num, 'is processing...', 'Current shape', x.shape, y.shape)

    np.save(file=output + "/test_x_FD001.npy", arr=x)
    np.save(file=output + "/test_y_FD001.npy", arr=y)
    np.save(file=output + "/valid_x_FD001.npy", arr=x_for_valid)
    print('The .npy file has been saved')
    '''


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ConvLSTM for RUL prediction on C-MAPSS')
    parser.add_argument('--tw', type=int, default=16, help='data processing parameter')
    parser.add_argument('--T', type=int, default=8, help='data processing parameter')
    args = parser.parse_args()
    input = "None"
    output = os.path.join(os.getcwd(), 'data', 'CMAPSSData', '%d_%d' % (args.tw, args.T))
    my_file = Path(output)
    if my_file.is_dir():
        pre_process(input, output)
    else:
        os.mkdir(my_file)
        pre_process(input, output)
