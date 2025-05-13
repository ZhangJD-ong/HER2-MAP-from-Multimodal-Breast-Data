import torch
from utils import confusion_matrix
from sklearn.metrics import roc_auc_score
import os
import torch.nn as nn
import pandas as pd
import utils
from config import config
import numpy as np
import random
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader,SubsetRandomSampler
from MODEL import Single_US,Single_MRI,Single_MM, Model3, CLIP
from hovertrans import create_model
from utils import confusion_matrix
import warnings

warnings.filterwarnings("ignore")

def valid(config, net, val_loader, criterion):
    device = next(net.parameters()).device
    net.eval()

    print("START VALIDING")
    epoch_loss = 0
    y_true, y_score = [], []
    cm = torch.zeros((config.class_num, config.class_num))

    for i, pack in enumerate(val_loader):

        MRI = pack['MRI'].type(torch.float32).to(device)
        US = pack['US'].type(torch.float32).to(device)
        MM = pack['MM'].type(torch.float32).to(device)
        clinical = pack['clinial'].type(torch.float32).to(device)
        modality_idx = pack['Modality_index'].type(torch.float32).to(device)
        labels = pack['labels'].type(torch.int64).to(device)
        names = pack['names']



        output,_,_,_,_,_,_ = net(MRI,US,MM,clinical,modality_idx)
        loss = criterion(output, labels)

        pred = output.argmax(dim=1)
        y_true.append(labels.detach().item())
        y_score.append(output[0].softmax(0)[1].item())

        cm = confusion_matrix(pred.detach(), labels.detach(), cm)
        epoch_loss += loss.cpu()

    avg_epoch_loss = epoch_loss / len(val_loader)

    acc = cm.diag().sum() / cm.sum()
    spe, sen = cm.diag() / (cm.sum(dim=1) + 1e-6)
    pre = cm.diag()[1] / (cm.sum(dim=0) + 1e-6)[1]
    rec = sen
    f1score = 2 * pre * rec / (pre + rec + 1e-6)
    auc = roc_auc_score(y_true, y_score)
    print('AUC:', auc, 'ACC:', acc, 'F1:', f1score, 'SEN:', sen, 'SPE:', spe, 'PRE:', pre)

    return [avg_epoch_loss, acc, sen, spe, auc, pre, f1score]


def seed_torch(seed=1):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

if __name__ == '__main__':
    seed_torch(42)
    args = config()
    test_set = utils.get_dataset(args.data_path, args.csv_path, args.clinical_path, args.img_size_US, args.img_size_MM,
                                 mode='test')

    print(args)
    argspath = os.path.join(args.model_path, args.model_name, args.writer_comment)

    if not os.path.exists(argspath):
        os.makedirs(argspath)
    with open(os.path.join(argspath, 'model_info.txt'), 'w') as f:
        f.write(str(args))
    All_data = pd.read_csv(os.path.join(args.csv_path))
    data_split = All_data['Train1Test2']
    test_idx = np.where(data_split ==2 )[0].tolist()

    print("\n")

    test_sampler = SubsetRandomSampler(test_idx)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, sampler=test_sampler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # MODEL
    model = Model3().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    ckpt = torch.load(os.path.join(args.model_path, args.model_name, args.writer_comment,'bestmodel.pth'), map_location=device)
    model.load_state_dict(ckpt)
    with torch.no_grad():
        valid(args, model, test_loader, criterion)