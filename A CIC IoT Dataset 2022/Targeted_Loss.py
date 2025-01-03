import numpy as np 
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import torch,os 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from preprocess import balance_classes, adjust_class_ratio
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable as V 
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_curve, precision_recall_curve,auc
from itertools import permutations

def configure_experiment_data(ood_label, target_ratio):
    feature = np.load("./data/X.npy").astype(np.float32)
    label = np.load("./data/y.npy")

    idx_ood = np.where(label==ood_label)[0].tolist()
    idx_in = list(set([i for i in range(feature.shape[0])]) - set(idx_ood))

    X_in = feature[idx_in,:]
    y_in = LabelEncoder().fit_transform(label[idx_in])

    X_ood = feature[idx_ood, :]

    X_train, X_0, y_train, _ = train_test_split(X_in, y_in, test_size=0.3)

    #print(X_0.shape, X_ood.shape, X_in.shape)
    X_test = np.concatenate([X_0, X_ood], axis=0)
    y_test = np.concatenate([
        np.zeros(X_0.shape[0]),np.ones(X_ood.shape[0])
    ])

    normalizer = StandardScaler()
    normalizer.fit(X_train)
    X_train = normalizer.transform(X_train)
    X_test = normalizer.transform(X_test)

    # adjust the distribution of train set and test set
    X_train, y_train = adjust_class_ratio(X_train, y_train, target_ratio)
    X_test, y_test = balance_classes(X_test, y_test)
    return X_train, y_train, X_test, y_test

class Classifier(nn.Module):
    def __init__(self, n_labels):
        super(Classifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(48, 36),
            nn.ReLU(),
            nn.Linear(36, n_labels)
        )
    def forward(self, x):
        x = x.view(-1, 48)
        av = self.net(x)
        return av

def targeted_loss(Z_batch, Y_batch, Anchors):
    pdist = torch.nn.PairwiseDistance(p=2)
    device = Z_batch.device
    if Anchors.device is not device:
        Anchors.to(device)
    
    our_loss = torch.tensor([0.0]).to(device)
    LA = torch.tensor([0.0]).to(device)
    
    for idx in range(Z_batch.shape[0]):
        cur_y = Y_batch.cpu().data.numpy()[idx]
        d = torch.tensor([
            pdist(Z_batch[idx].reshape(1,-1), Anchors[jdx].reshape(1,-1)) for jdx in range(Anchors.shape[0])
        ])
        items_to_add = [torch.exp(d[cur_y]-d[jdx]) for jdx in range(Anchors.shape[0])]
        _loss_cur = torch.tensor([0.0]).to(device)
        for item in items_to_add:
            _loss_cur = torch.add(_loss_cur, item)
        _loss_cur = torch.log2(_loss_cur)
        our_loss = torch.add(our_loss, _loss_cur)
        LA = torch.add(LA, pdist(Z_batch[idx].reshape(1,-1), Anchors[cur_y].reshape(1,-1)))
    LA = LA / Z_batch.shape[0]
    our_loss = our_loss / Z_batch.shape[0]
    return our_loss + 0.05*LA

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


def train_process(X_train, y_train):
    seed_value = 43
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    
    tensor_X_train = torch.from_numpy(X_train.astype(np.float32))
    tensor_y_train = torch.from_numpy(LabelEncoder().fit_transform(y_train))
    my_dataset = TensorDataset(tensor_X_train, tensor_y_train)
    my_dataloader = DataLoader(my_dataset, batch_size=40, shuffle=True)
    
    EPOCH = 40
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Classifier(
        len(np.unique(y_train))
    ).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    Anchors = torch.tensor(np.diag([1 for i in range(len(np.unique(y_train)))])).to(DEVICE)
    
    for epoch in tqdm(range(EPOCH), desc='train process'):
        total_loss = 0
        for i, (x,y) in enumerate(my_dataloader):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            av = model(V(x))
            loss = targeted_loss(av,y,Anchors)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
        if epoch % 5 == 0:
            print('Epoch :', epoch, '|', f'train_loss:{total_loss.data}')
            
    centers = calculate_centers(model, X_train, y_train)
    scores_of_train_samples = calculate_novelty_score(model, centers, X_train)
    threshold = np.quantile(scores_of_train_samples, 0.95)
    return model, threshold, centers

def test_process(model, threshold,  centers, X_test, y_test):
    novelty_scores = calculate_novelty_score(model, centers, X_test)
    
    fprs, tprs, thres_roc = roc_curve(y_test, novelty_scores, pos_label=1)
    auroc_value = auc(fprs, tprs)
    
    precisions, recalls, thres_pr = precision_recall_curve(y_test, novelty_scores)
    aupr_value = auc(recalls, precisions)
    
    y_pred = (novelty_scores > threshold).astype('int')
    macro_f1_score_value = f1_score(y_test, y_pred, average='macro')
    
    fpr95 = np.min([1 - p for p, r in zip(precisions, recalls) if r >= 0.95])
    return auroc_value, aupr_value, macro_f1_score_value, fpr95


if __name__ == "__main__":
    ood_label_pool = ['Audio','Camera','Home Automation']
    for ood_label in ood_label_pool:
        result_file_name = f'{ood_label}'
        r_auroc = []
        r_aupr = []
        r_f1 = []
        r_fpr95 = []
        r_ratio = []
        #for target_ratio in [1,2, 5,10,30,50,70,100,300,500,700,1000]:
        for target_ratio in [1,2,5,10,15,20,25,50,70,100]:
            print(f'evaluation when {ood_label} as ood label','|', f'ratio of train set:{target_ratio}')
            X_train, y_train, X_test, y_test = configure_experiment_data(ood_label, target_ratio)
            model, threshold, centers = train_process(X_train, y_train)
            auroc_value, aupr_value, macro_f1_score_value, fpr95_value = test_process(model, threshold, centers, X_test, y_test)
            print(f'auroc:{auroc_value}', '|', 
                  f'aupr:{aupr_value}', '|', 
                  f'macro f1 score:{macro_f1_score_value}','|',
                  f'fpr95:{fpr95_value}') 
            r_auroc.append(auroc_value)
            r_aupr.append(aupr_value)
            r_f1.append(macro_f1_score_value)
            r_fpr95.append(fpr95_value)
            r_ratio.append(target_ratio)
        
        result = {
            'ratio':r_ratio,
            'auroc':r_auroc,
            'aupr':r_aupr,
            'macro f1 score':r_f1,
            'fpr95':r_fpr95
        }
        pd.DataFrame(result).to_csv(f'./result/{result_file_name}_Targeted_Loss.csv')
