import numpy as np 
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable as V 
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from itertools import permutations
from sklearn.metrics import f1_score, roc_curve, precision_recall_curve,auc
from collections import Counter 

def calculate_proportions_and_variance(arr):
    counter = Counter(arr)
    total_count = len(arr)
    proportions = {key:value / total_count for key, value in counter.items()}
    proportions_values = np.array(
        list(proportions.values())
    )
    variance = np.var(proportions_values)
    return variance


class Classifier(nn.Module):
    def __init__(self, n_labels):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(76, 32),
            nn.ReLU(),
            nn.Linear(32, n_labels)
        )
        self.centers = None
    def forward(self, x):
        x = x.view(-1, 76)
        av = self.net(x)
        return av


def ii_loss(Z_batch, Y_batch):
    """
    功能：
        提取当前epoch样本的所有标签
        计算每个标签对应的激活向量的均值中心
        计算每个激活向量到对应均值中心的距离
        最小化：激活向量到对应均值中心的距离之和 / 标签个数
        最大化：不同均值中心间距离的最小值
    """
    pdist = torch.nn.PairwiseDistance(p=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    label_value = np.unique(Y_batch.data.cpu().numpy()).tolist()
    #loss_intra_spread = torch.tensor([0.0]).requires_grad_(True).to(device)
    loss_intra_spread = torch.tensor([0.0]).to(device)
    mean_of_all_label = []
    
    for label in label_value:
        # 当前标签对应的激活向量在当前epoch中的位置
        indices = torch.from_numpy(np.where(Y_batch.data.cpu().numpy()==label)[0]).to(device)
        # 找到当前标签在当前epoch中的那些激活向量
        cur_label_tensor = torch.index_select(Z_batch, 0, indices)
        # 计算当前标签在当前epoch中的那些激活向量的均值中心
        mean_of_cur_label = torch.mean(cur_label_tensor, dim=0)
        mean_of_all_label.append(mean_of_cur_label)

        # 计算当前标签在当前epoch中的激活向量到对应的均值中心之间的距离
        for idx in range(cur_label_tensor.shape[0]):
            dis = pdist(mean_of_cur_label.reshape(1,-1), cur_label_tensor[idx].reshape(1,-1))
            loss_intra_spread += dis
    
    loss_intra_spread = loss_intra_spread / Z_batch.shape[0]

    # 计算当前epoch中不同标签的均值中心之间欧式距离的最小值
    center_pairs = list(permutations([i for i in range(len(mean_of_all_label))], 2))
    distance_between_centers = [
        pdist(mean_of_all_label[a].reshape(1,-1), mean_of_all_label[b].reshape(1,-1)).cpu().data.numpy() for (a,b) in center_pairs
    ]
    #print(distance_between_centers)
    loss_inter_sparation = np.min(distance_between_centers)
    return loss_intra_spread - loss_inter_sparation

def calculate_centers(model, X_train, y_train):
    device = next(model.parameters()).device
    tensor_X = torch.from_numpy(X_train.astype(np.float32)).to(device)
    embedding = model(tensor_X)
    
    y_pred = np.argmax(F.softmax(embedding, dim=0).data.cpu().numpy(), axis=1)
    
    _centers = {}
    for label in np.unique(y_train).tolist():
        _centers[label] = np.zeros(embedding.shape[1])
        _centers[label][label] = 1
        
    chosen_embedding = embedding.data.cpu().numpy()[np.where(y_train==y_pred)[0].tolist(), :]
    chosen_label = y_train[np.where(y_train==y_pred)[0].tolist()]
    
    for label in np.unique(y_train).tolist():
        idx = np.where(chosen_label==label)[0].tolist()
        if len(idx) == 0:
            _centers[label] = np.mean(embedding.cpu().detach().numpy()[np.where(y_train==label)[0].tolist(), :], axis=0)
        else:
            _centers[label] = np.mean(chosen_embedding[idx, :], axis=0)
    
    return _centers

def calculate_novelty_score(model, centers, X):
    device = next(model.parameters()).device
    tensor_X = torch.from_numpy(X.astype(np.float32))
    _scores = np.zeros(X.shape[0])
    Z = model(V(tensor_X.to(device)))

    for i in range(X.shape[0]):
        distances = np.array([
            np.linalg.norm(centers[k]-Z[i].reshape(1,-1).data.cpu().numpy()) for k in centers
        ])
        _scores[i] = np.min(distances)
    return _scores


def train_process(X_train, y_train, X_eval, y_eval):
    seed_value = 43
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    
    tensor_X_train = torch.from_numpy(X_train.astype(np.float32))
    tensor_y_train = torch.from_numpy(y_train)
    my_dataset = TensorDataset(tensor_X_train, tensor_y_train)
    my_dataloader = DataLoader(my_dataset, batch_size=40, shuffle=True)
    
    EPOCH = 100
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Classifier(
        len(np.unique(y_train))
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    PI = 0.005
    
    for epoch in tqdm(range(EPOCH), desc='train process'):
        total_loss = 0
        for i, (x,y) in enumerate(my_dataloader):
            if len(np.unique(y.numpy())) < 2:
                continue
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            av = model(V(x))
            loss = criterion(av,y) + PI*ii_loss(av, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
        if epoch % 5 == 0:
            print('Epoch :', epoch, '|', f'train_loss:{total_loss.data}')
    
    model.eval()
    FPR_list = np.arange(0.001, 0.05, 0.002)
    centers = calculate_centers(model, X_train, y_train)
    scores_all = calculate_novelty_score(model, centers, X_eval)
    scores_neg = scores_all[y_eval == 0]
    best_threshold, best_score = None, 0
    for FPR in FPR_list:
        threshold = np.quantile(scores_neg, 1-FPR)
        y_pred = (scores_all > threshold).astype('int')
        score = f1_score(y_eval, y_pred, average='macro')
        if score > best_score:
            #print('FPR:', FPR, 'f1_score:', score)
            best_threshold = threshold
            best_score = score
    
    return model, best_threshold, centers

def test_process(model, threshold, X_test, y_test, centers):
    novelty_scores = calculate_novelty_score(model, centers, X_test)
    
    fprs, tprs, thres_roc = roc_curve(y_test, novelty_scores, pos_label=1)
    auroc_value = auc(fprs, tprs)
    
    precisions, recalls, thres_pr = precision_recall_curve(y_test, novelty_scores)
    aupr_value = auc(recalls, precisions)
    
    y_pred = (novelty_scores > threshold).astype('int')
    macro_f1_score_value = f1_score(y_test, y_pred, average='macro')
    fpr95 = np.min([1 - p for p, r in zip(precisions, recalls) if r >= 0.95])
    return auroc_value, aupr_value, macro_f1_score_value, fpr95, novelty_scores

if __name__ == "__main__":
    print("experiment using Power and Idle")
    
    X_train = np.load("./data/Idle/X_train.npy").astype(np.float32)
    y_train = np.load("./data/Idle/y_train.npy")
    X_eval = np.load("./data/Idle/X_eval.npy").astype(np.float32)
    y_eval = np.load("./data/Idle/y_eval.npy")
    X_test = np.load("./data/Idle/X_test.npy").astype(np.float32)
    y_test = np.load("./data/Idle/y_test.npy")
    
    model, best_threshold, centers = train_process(X_train, y_train, X_eval, y_eval)
    auroc_value, aupr_value, macro_f1_score_value, fpr95_value, novelty_scores = test_process(model, best_threshold, X_test, y_test, centers)
    np.save("./result/Idle/II_Loss_score", novelty_scores.astype(np.float32))
    print(f'auroc:{auroc_value}', '|', f'aupr:{aupr_value}', '|', f'macro f1 score:{macro_f1_score_value}','|',f'fpr95:{fpr95_value}')
    # auroc:0.6297650114247899 | aupr:0.6041414713234067 | macro f1 score:0.4334191222215673 | fpr95:0.48121387283236994
    
    print("experiment using Power and Attack")
    
    X_train = np.load("./data/Attack/X_train.npy").astype(np.float32)
    y_train = np.load("./data/Attack/y_train.npy")
    X_eval = np.load("./data/Attack/X_eval.npy").astype(np.float32)
    y_eval = np.load("./data/Attack/y_eval.npy")
    X_test = np.load("./data/Attack/X_test.npy").astype(np.float32)
    y_test = np.load("./data/Attack/y_test.npy")
    
    model, best_threshold, centers = train_process(X_train, y_train, X_eval, y_eval)
    auroc_value, aupr_value, macro_f1_score_value, fpr95_value, novelty_scores = test_process(model, best_threshold, X_test, y_test, centers)
    np.save("./result/Attack/II_Loss_score", novelty_scores.astype(np.float32))
    print(f'auroc:{auroc_value}', '|', f'aupr:{aupr_value}', '|', f'macro f1 score:{macro_f1_score_value}','|',f'fpr95:{fpr95_value}')
    # auroc:0.7211414801033851 | aupr:0.7358579594518346 | macro f1 score:0.593172657614901 | fpr95:0.4843304843304843
    
    