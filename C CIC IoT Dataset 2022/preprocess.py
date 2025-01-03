import numpy as np 
import pandas as pd 
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def read_csv_CIC_IoT_Dataset_2022(filepath):
    df = pd.read_csv(filepath, index_col=1,encoding='unicode-escape')
    df.replace(np.inf, np.nan, inplace=True)
    df = df.dropna(how='any', axis=0)
    columns_unused = [
        'Flow ID',
        'Src IP',
        'Src Port',
        'Dst IP',
        'Dst Port',
        'Protocol',
        'Timestamp']
    df = df[list(set(df.columns) - set(columns_unused))]
    feature = df[list(set(df.columns)-set(['Label']))].to_numpy()
    return feature

# 提取Audio类型的CIC Flowmeter特征
# 提取Camera类型的CIC Flowmeter特征
# 提取Home Automation类型的CIC Flowmeter特征
def extract_from_CIC_IoT_Dataset_2022_Power():
    all_features = []
    all_labels = []
    
    #data_dir = "/root/dataset/CIC_Flowmeter_feature_CIC_IoT_Dataset2022/Power"
    data_dir = "D:\\dataset\\CIC_Flowmeter_feature_CIC_IoT_Dataset2022\\Power"
    obj_function = [item for item in os.scandir(data_dir)]
    for obj in obj_function:
        obj_device = [item for item in os.scandir(obj.path)]
        for device in obj_device:
            _csvs = os.scandir(device.path)
            for csv_file in _csvs:
                feature = read_csv_CIC_IoT_Dataset_2022(csv_file.path)
                label = np.array([obj.name]*feature.shape[0])
                all_features.append(feature)
                all_labels.append(label)
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels)
    return features, labels

# 提取Idle阶段的CIC Flowmeter特征
def extract_from_CIC_IoT_Dataset_2022_Idle():
    all_features = []
    all_labels = []
    
    #data_dir = "/root/dataset/CIC_Flowmeter_feature_CIC_IoT_Dataset2022/Idle/csvs"
    data_dir = "D:\\dataset\\CIC_Flowmeter_feature_CIC_IoT_Dataset2022\\Idle\\csvs"
    csv_files = [item for item in os.scandir(data_dir)]
    for _csv in csv_files:
        feature = read_csv_CIC_IoT_Dataset_2022(_csv.path)
        name = _csv.name.split('.')[0]
        label = np.array([name]*feature.shape[0])
        all_features.append(feature)
        all_labels.append(label)
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels)
    return features, labels

# 提取Attack阶段的CIC Flowmeter特征
def extract_from_CIC_IoT_Dataset_2022_Attack():
    
    def exchange_label_name(raw_label):
        new_label = None
        if 'HTTPFlood' in raw_label:
            new_label = 'HTTPFlood'
        if 'TCPFlood' in raw_label:
            new_label = 'TCPFlood'
        if 'RTSP' in raw_label:
            new_label = 'RTSP'
        if 'RSTP' in raw_label:
            new_label = 'RSTP'
        if 'UDPFlood' in raw_label:
            new_label = 'UDPFlood'
        if 'BruteForce' in raw_label:
            new_label = 'BruteForce'
        
        return new_label
    
    all_features = []
    all_labels = []
    
    #data_dir = "/root/dataset/CIC_Flowmeter_feature_CIC_IoT_Dataset2022/Attack"
    data_dir = "D:\\dataset\\CIC_Flowmeter_feature_CIC_IoT_Dataset2022\\Attack"
    
    csv_files = [item for item in os.scandir(data_dir)]
    for _csv in csv_files:
        feature = read_csv_CIC_IoT_Dataset_2022(_csv.path)
        name = exchange_label_name(_csv.name.split('_')[0])
        label = np.array([name]*feature.shape[0])
        all_features.append(feature)
        all_labels.append(label)
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels)
    return features, labels



def balance_classes(features, labels):
    """
    重新采样，使得两种标签的样本数量一致。

    :param features: 二维numpy数组，特征集
    :param labels: 一维numpy数组，标签集
    :return: 重新采样后的特征集和标签集
    """
    # 找出两种标签的样本数量
    unique, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique, counts))
    
    if len(label_counts) != 2:
        raise ValueError("标签数量不等于2，无法平衡样本数量。")
    
    label_1, label_2 = unique
    count_1, count_2 = label_counts[label_1], label_counts[label_2]
    
    # 找出数量较少的标签和数量较多的标签
    if count_1 < count_2:
        minority_label, majority_label = label_1, label_2
        minority_count, majority_count = count_1, count_2
    else:
        minority_label, majority_label = label_2, label_1
        minority_count, majority_count = count_2, count_1
    
    # 获取少数类和多数类的样本索引
    minority_indices = np.where(labels == minority_label)[0]
    majority_indices = np.where(labels == majority_label)[0]
    
    # 重新采样多数类样本，使其数量与少数类样本一致
    np.random.shuffle(majority_indices)
    sampled_majority_indices = majority_indices[:minority_count]
    
    # 合并少数类和重新采样后的多数类样本
    new_indices = np.concatenate([minority_indices, sampled_majority_indices])
    
    # 返回重新采样后的特征集和标签集
    return features[new_indices], labels[new_indices]



# 切分训练集、测试集、验证集
if __name__ == "__main__":
    X_Power, y_Power = extract_from_CIC_IoT_Dataset_2022_Power()
    
    X_Idle, _ = extract_from_CIC_IoT_Dataset_2022_Idle()
    y_Idle = np.ones(X_Idle.shape[0])
    
    X_Attack, _ = extract_from_CIC_IoT_Dataset_2022_Attack()
    y_Attack = np.ones(X_Attack.shape[0])
    
    print(X_Power.shape)
    print(X_Idle.shape)
    print(X_Attack.shape)    
    
    X_Power_train, X_Power_, y_Power_train, _ = train_test_split(
        X_Power, y_Power, test_size=0.4
    )
    y_Power_ = np.zeros(X_Power_.shape[0])
    
    X_Power_eval, X_Power_test, y_Power_eval, y_Power_test = train_test_split(
        X_Power_, y_Power_, test_size=0.5
    )
    
    # Power and Idle
    X_Idle_eval, X_Idle_test, y_Idle_eval, y_Idle_test = train_test_split(
        X_Idle, y_Idle, test_size=0.5
    )
    
    X_train = X_Power_train
    y_train = LabelEncoder().fit_transform(y_Power_train)
    
    X_eval = np.concatenate([X_Power_eval, X_Idle_eval], axis=0)
    y_eval = np.concatenate([y_Power_eval, y_Idle_eval])
    X_eval, y_eval = balance_classes(X_eval, y_eval)
    
    X_test = np.concatenate([X_Power_test, X_Idle_test], axis=0)
    y_test = np.concatenate([y_Power_test, y_Idle_test])
    X_test, y_test = balance_classes(X_test, y_test)
    
    normalizer = StandardScaler()
    normalizer = normalizer.fit(X_train)
    X_train = normalizer.transform(X_train)
    X_eval = normalizer.transform(X_eval)
    X_test = normalizer.transform(X_test)
    
    print(pd.DataFrame(y_train).value_counts())
    print(pd.DataFrame(y_eval).value_counts())
    print(pd.DataFrame(y_test).value_counts())
    
    np.save("./data/Idle/X_train.npy", X_train.astype(np.float32))
    np.save("./data/Idle/y_train.npy", y_train)
    np.save("./data/Idle/X_eval.npy", X_eval.astype(np.float32))
    np.save("./data/Idle/y_eval.npy", y_eval)
    np.save("./data/Idle/X_test.npy", X_test.astype(np.float32))
    np.save("./data/Idle/y_test.npy", y_test)
    
    # Power and Attack
    X_Attack_eval, X_Attack_test, y_Attack_eval, y_Attack_test = train_test_split(
        X_Attack, y_Attack, test_size=0.5
    )
    
    X_train = X_Power_train
    y_train = LabelEncoder().fit_transform(y_Power_train)
    
    X_eval = np.concatenate([X_Power_eval, X_Attack_eval], axis=0)
    y_eval = np.concatenate([y_Power_eval, y_Attack_eval])
    X_eval, y_eval = balance_classes(X_eval, y_eval)
    
    X_test = np.concatenate([X_Power_test, X_Attack_test], axis=0)
    y_test = np.concatenate([y_Power_test, y_Attack_test])
    X_test, y_test = balance_classes(X_test, y_test)
    
    normalizer = StandardScaler()
    normalizer = normalizer.fit(X_train)
    X_train = normalizer.transform(X_train)
    X_eval = normalizer.transform(X_eval)
    X_test = normalizer.transform(X_test)
    
    print(pd.DataFrame(y_train).value_counts())
    print(pd.DataFrame(y_eval).value_counts())
    print(pd.DataFrame(y_test).value_counts())
    
    np.save("./data/Attack/X_train.npy", X_train.astype(np.float32))
    np.save("./data/Attack/y_train.npy", y_train)
    np.save("./data/Attack/X_eval.npy", X_eval.astype(np.float32))
    np.save("./data/Attack/y_eval.npy", y_eval)
    np.save("./data/Attack/X_test.npy", X_test.astype(np.float32))
    np.save("./data/Attack/y_test.npy", y_test)