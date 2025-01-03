import numpy as np 
import pandas as pd 

import os

RAW_CSV_DIR = "/root/dataset/CIC_IOT_Dataset2022/CSV files/CIC Device Type"

def self_read_csv(filepath):
    df = pd.read_csv(filepath, index_col=1)
    df.replace(np.inf, np.nan, inplace=True)
    df.dropna(how='any', axis=0, inplace=True)
    return df

def extract_audio():
    function_type = 'Audio'
    working_dir = os.path.join(RAW_CSV_DIR, function_type)
    np_features = []
    np_labels = []
    
    modes = [item for item in os.scandir(working_dir) if os.path.isdir(item)]
    
    for mode in modes:
        data_paths = [
            item.path for item in os.scandir(mode.path) if item.name.split(".")[-1]=='csv'
        ]
        if len(data_paths) > 0:
            label = mode.name
            tmp_df = pd.concat(
                [self_read_csv(filepath) for filepath in data_paths],
                axis=0
            )
            label = np.array([label]*tmp_df.shape[0])
            feature = tmp_df.to_numpy()
            np_features.append(feature)
            np_labels.append(label)
    all_features = np.concatenate(np_features, axis=0)
    all_labels = np.concatenate(np_labels,axis=0)
    
    return all_features


def extract_camera():
    function_type = 'Camera'
    working_dir = os.path.join(RAW_CSV_DIR, function_type)
    np_features = []
    np_labels = []
    
    modes = [item for item in os.scandir(working_dir) if os.path.isdir(item)]
    
    for mode in modes:
        data_paths = [
            item.path for item in os.scandir(mode.path) if item.name.split(".")[-1]=='csv'
        ]
        if len(data_paths) > 0:
            label = mode.name
            tmp_df = pd.concat(
                [self_read_csv(filepath) for filepath in data_paths],
                axis=0
            )
            label = np.array([label]*tmp_df.shape[0])
            feature = tmp_df.to_numpy()
            np_features.append(feature)
            np_labels.append(label)
    all_features = np.concatenate(np_features, axis=0)
    all_labels = np.concatenate(np_labels,axis=0)
    
    return all_features


def extract_home_automation():
    function_type = 'Home Automation'
    working_dir = os.path.join(RAW_CSV_DIR, function_type)
    np_features = []
    np_labels = []
    
    modes = [item for item in os.scandir(working_dir) if os.path.isdir(item)]
    
    for mode in modes:
        data_paths = [
            item.path for item in os.scandir(mode.path) if item.name.split(".")[-1]=='csv'
        ]
        if len(data_paths) > 0:
            label = mode.name
            tmp_df = pd.concat(
                [self_read_csv(filepath) for filepath in data_paths],
                axis=0
            )
            label = np.array([label]*tmp_df.shape[0])
            feature = tmp_df.to_numpy()
            np_features.append(feature)
            np_labels.append(label)
    all_features = np.concatenate(np_features, axis=0)
    all_labels = np.concatenate(np_labels,axis=0)
    
    return all_features
    
    

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

def adjust_train_set_distribution(features, labels, target_ratio):
    """
    调整少数类的样本数量，使得多数类和少数类的样本数量的比值符合指定的参数。

    :param features: 二维numpy数组，特征集
    :param labels: 一维numpy数组，标签集
    :param target_ratio: 多数类与少数类样本数量的目标比值，float类型
    :return: 调整后的特征集和标签集
    """
    # 找出多数类和少数类
    unique, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique, counts))
    
    majority_label = max(label_counts, key=label_counts.get)
    minority_label = min(label_counts, key=label_counts.get)
    
    majority_count = label_counts[majority_label]
    minority_count = label_counts[minority_label]
    
    # 计算需要的少数类样本数量
    required_minority_count = int(majority_count / target_ratio)
    
    # 检查是否满足目标比值
    if required_minority_count > minority_count:
        current_ratio = majority_count / minority_count
        raise ValueError(f"无法满足目标比值。当前多数类样本数量: {majority_count}, 少数类样本数量: {minority_count}, 当前比值: {current_ratio:.2f}")
    
    # 获取多数类和少数类的样本索引
    majority_indices = np.where(labels == majority_label)[0]
    minority_indices = np.where(labels == minority_label)[0]
    
    # 采样少数类样本
    np.random.shuffle(minority_indices)
    sampled_minority_indices = minority_indices[:required_minority_count]
    
    # 合并多数类和少数类样本
    new_indices = np.concatenate([majority_indices, sampled_minority_indices])
    
    # 返回调整后的特征集和标签集
    return features[new_indices], labels[new_indices]



def adjust_class_ratio(features, labels, ratio):
    """
    调整多数类和少数类的样本数量，使得它们的比值符合指定的参数ratio。

    :param features: 二维numpy数组，特征集
    :param labels: 一维numpy数组，标签集
    :param ratio: 目标比值，float类型
    :return: 调整后的特征集和标签集
    """
    # 找出多数类和少数类
    unique, counts = np.unique(labels, return_counts=True)
    label_counts = dict(zip(unique, counts))
    
    if len(label_counts) != 2:
        raise ValueError("标签数量不等于2，无法调整样本数量比值。")
    
    label_1, label_2 = unique
    count_1, count_2 = label_counts[label_1], label_counts[label_2]
    
    # 找出多数类和少数类
    if count_1 > count_2:
        majority_label, minority_label = label_1, label_2
        majority_count, minority_count = count_1, count_2
    else:
        majority_label, minority_label = label_2, label_1
        majority_count, minority_count = count_2, count_1
    
    # 计算当前比值
    current_ratio = majority_count / minority_count
    
    # 根据目标比值调整样本数量
    if ratio < current_ratio:
        # 从多数类中采样部分样本
        required_majority_count = int(minority_count * ratio)
        majority_indices = np.where(labels == majority_label)[0]
        np.random.shuffle(majority_indices)
        sampled_majority_indices = majority_indices[:required_majority_count]
        minority_indices = np.where(labels == minority_label)[0]
        new_indices = np.concatenate([sampled_majority_indices, minority_indices])
    else:
        # 从少数类中采样部分样本
        required_minority_count = int(majority_count / ratio)
        minority_indices = np.where(labels == minority_label)[0]
        np.random.shuffle(minority_indices)
        sampled_minority_indices = minority_indices[:required_minority_count]
        majority_indices = np.where(labels == majority_label)[0]
        new_indices = np.concatenate([majority_indices, sampled_minority_indices])
    
    # 返回调整后的特征集和标签集
    return features[new_indices], labels[new_indices]



if __name__ == "__main__":
    # 合并三种设备的特征 并打标签
    # 保存设备流量特征为numpy数组并保存
    # 保存设备标签为numpy数组并保存
    X_audio = extract_audio()
    y_audio = np.array(['Audio']*X_audio.shape[0])
    # 18774
    
    X_camera = extract_camera()
    y_camera = np.array(['Camera']*X_camera.shape[0])
    #191175
    
    X_home_automation = extract_home_automation()
    y_home_automation = np.array(['Home Automation']*X_home_automation.shape[0])
    # 19808
    
    X = np.concatenate([X_audio, X_camera, X_home_automation], axis=0)
    y = np.concatenate([y_audio, y_camera, y_home_automation])

    print(np.unique(y))
    
    np.save("./data/X.npy", X.astype(np.float32))
    np.save("./data/y.npy", y)