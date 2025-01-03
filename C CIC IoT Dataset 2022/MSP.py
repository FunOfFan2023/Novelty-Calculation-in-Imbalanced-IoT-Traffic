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
from sklearn.metrics import f1_score, roc_curve, precision_recall_curve,auc
from collections import Counter 

class Classifier(nn.Module):
    def __init__(self, n_labels):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(76, 32),
            nn.ReLU(),
            nn.Linear(32, n_labels)
        )
    def forward(self, x):
        x = x.view(-1, 76)
        av = self.net(x)
        return av

def calculate_novelty_score(model, X):
    _scores = np.zeros(X.shape[0])
    model.eval()
    DEVICE = next(model.parameters()).device
    tensor_X = torch.from_numpy(X.astype(np.float32)).to(DEVICE)
    av = model(V(tensor_X))
    softmax_av = F.softmax(av, dim=0).cpu().detach().numpy()
    for i in range(softmax_av.shape[0]):
        _scores[i] = np.max(softmax_av[i,:])
    _scores = np.exp(-_scores)
    return _scores

def train_process(X_train, y_train, X_eval, y_eval):
    seed_value = 43
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    
    tensor_X_train = torch.from_numpy(X_train.astype(np.float32))
    tensor_y_train = torch.from_numpy(LabelEncoder().fit_transform(y_train))
    my_dataset = TensorDataset(tensor_X_train, tensor_y_train)
    my_dataloader = DataLoader(my_dataset, batch_size=40, shuffle=True)
    
    EPOCH = 150
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Classifier(
        len(np.unique(y_train))
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    
    for epoch in tqdm(range(EPOCH), desc='train process'):
        total_loss = 0
        for i, (x,y) in enumerate(my_dataloader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            av = model(V(x))
            loss = criterion(av,y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
        if epoch % 10 == 0:
            print('Epoch :', epoch, '|', f'train_loss:{total_loss.data}')
    
    model.eval()
    FPR_list = np.arange(0.001, 0.05, 0.002)
    scores_all = calculate_novelty_score(model, X_eval)
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
    
    return model, best_threshold

def test_process(model, threshold, X_test, y_test):
    novelty_scores = calculate_novelty_score(model, X_test)
    
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
    
    model, best_threshold = train_process(X_train, y_train, X_eval, y_eval)
    auroc_value, aupr_value, macro_f1_score_value, fpr95_value, novelty_scores = test_process(model, best_threshold, X_test, y_test)
    np.save("./result/Idle/MSP_score", novelty_scores.astype(np.float32))
    print(f'auroc:{auroc_value}', '|', f'aupr:{aupr_value}', '|', f'macro f1 score:{macro_f1_score_value}','|',f'fpr95:{fpr95_value}')
    # auroc:0.5305460955936521 | aupr:0.5553930150645343 | macro f1 score:0.4648227509186077 | fpr95:0.4997788589119858
    
    print("experiment using Power and Attack")
    
    X_train = np.load("./data/Attack/X_train.npy").astype(np.float32)
    y_train = np.load("./data/Attack/y_train.npy")
    X_eval = np.load("./data/Attack/X_eval.npy").astype(np.float32)
    y_eval = np.load("./data/Attack/y_eval.npy")
    X_test = np.load("./data/Attack/X_test.npy").astype(np.float32)
    y_test = np.load("./data/Attack/y_test.npy")
    
    model, best_threshold = train_process(X_train, y_train, X_eval, y_eval)
    auroc_value, aupr_value, macro_f1_score_value, fpr95_value, novelty_scores = test_process(model, best_threshold, X_test, y_test)
    np.save("./result/Attack/MSP_score", novelty_scores.astype(np.float32))
    print(f'auroc:{auroc_value}', '|', f'aupr:{aupr_value}', '|', f'macro f1 score:{macro_f1_score_value}','|',f'fpr95:{fpr95_value}')
    # auroc:0.6792061019615677 | aupr:0.6940929472488959 | macro f1 score:0.3333333333333333 | fpr95:0.49955791335101685
    