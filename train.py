import torch
import pandas as pd
import os
import torch.nn as nn
import torchvision.models
import utils
from config import config
import numpy as np
import random
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.sampler import WeightedRandomSampler
from valid import valid
from MODEL import Single_US, Single_MRI, Single_MM, Model3, CLIP
from utils import confusion_matrix
import math
import warnings

warnings.filterwarnings("ignore")


def train(config, train_loader, test_loader, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    # MODEL
    model = Model3().to(device)

    model.train()
    align_loss = nn.L1Loss()

    if config.loss_function == 'CE':
        criterion = nn.CrossEntropyLoss().to(device)

    if config.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    elif config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    elif config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=5e-4)

    if config.scheduler == 'cosine':
        lr_lambda = lambda epoch: (epoch * (1 - config.warmup_decay) / config.warmup_epochs + config.warmup_decay) \
            if epoch < config.warmup_epochs else \
            (1 - config.min_lr / config.lr) * 0.5 * (math.cos((epoch - config.warmup_epochs) / (
                    config.epochs - config.warmup_epochs) * math.pi) + 1) + config.min_lr / config.lr
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)
    elif config.scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step, gamma=0.9)

    writer = SummaryWriter(comment='_' + config.model_name + '_' + config.writer_comment)

    print("START TRAINING")
    best_acc = 0
    ckpt_path = os.path.join(config.model_path, config.model_name, config.writer_comment)
    model_save_path = os.path.join(ckpt_path)
    cm = None

    for epoch in range(config.epochs):
        print("=======Epoch:{}======Learning_rate:{}=========".format(epoch + 1, optimizer.param_groups[0]['lr']))
        cm = torch.zeros((config.class_num, config.class_num))
        epoch_loss = 0
        for i, pack in enumerate(train_loader):

            MRI = pack['MRI'].type(torch.float32).to(device)
            US = pack['US'].type(torch.float32).to(device)
            MM = pack['MM'].type(torch.float32).to(device)
            clinical = pack['clinial'].type(torch.float32).to(device)
            modality_idx = pack['Modality_index'].to(device)
            labels = pack['labels'].to(device)
            names = pack['names']

            output, mri_out, us_out, mm_out, mri, us, mm = model(MRI, US, MM, clinical, modality_idx)
            w_mri_us = (modality_idx[:, 0] * modality_idx[:, 1]).unsqueeze(1)
            w_mri_mm = (modality_idx[:, 2] * modality_idx[:, 1]).unsqueeze(1)
            w_mm_us = (modality_idx[:, 0] * modality_idx[:, 2]).unsqueeze(1)
            Align_loss = (align_loss(mri*w_mri_us,us*w_mri_us)+align_loss(mri*w_mri_mm,mm*w_mri_mm)+align_loss(mm*w_mm_us,us*w_mm_us))/3

            loss = criterion(output, labels)
            +criterion(mri_out * (modality_idx[:, 1].unsqueeze(1)), labels * modality_idx[:, 1])
            +criterion(us_out * (modality_idx[:, 0].unsqueeze(1)), labels * modality_idx[:, 0])
            +criterion(mm_out * (modality_idx[:, 2].unsqueeze(1)), labels * modality_idx[:, 2])+1.0*Align_loss

            pred = output.argmax(dim=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            cm = confusion_matrix(pred.detach(), labels.detach(), cm)
        lr_scheduler.step()

        if (epoch + 1) % config.log_step == 0:
            print('[epoch %d]' % epoch)
            with torch.no_grad():
                result = valid(config, model, val_loader, criterion)
            val_loss, val_acc, sen, spe, auc, pre, f1score = result
            writer.add_scalar('Val/F1score', f1score, global_step=epoch)
            writer.add_scalar('Val/Pre', pre, global_step=epoch)
            writer.add_scalar('Val/Spe', spe, global_step=epoch)
            writer.add_scalar('Val/Sen', sen, global_step=epoch)
            writer.add_scalar('Val/AUC', auc, global_step=epoch)
            writer.add_scalar('Val/Acc', val_acc, global_step=epoch)
            writer.add_scalar('Val/Val_loss', val_loss, global_step=epoch)

            if epoch > config.epochs // 10:
                if auc > best_acc:
                    best_acc = auc
                    print("=> saved best model")
                    if not os.path.exists(model_save_path):
                        os.makedirs(model_save_path)
                    if config.save_model:
                        torch.save(model.state_dict(), os.path.join(model_save_path, 'bestmodel.pth'))
                    with open(os.path.join(model_save_path, 'result.txt'), 'w') as f:
                        f.write('Best Result:\n')
                        f.write('Acc: %f, Spe: %f, Sen: %f, AUC: %f, Pre: %f, F1score: %f'
                                % (val_acc, spe, sen, auc, pre, f1score))
        if epoch + 1 == config.epochs:
            with torch.no_grad():
                result = valid(config, model, test_loader, criterion)
            val_loss, val_acc, sen, spe, auc, pre, f1score = result
            if config.save_model:
                torch.save(model.state_dict(), os.path.join(model_save_path, 'last_epoch_model.pth'))
            with open(os.path.join(model_save_path, 'result.txt'), 'a') as f:
                f.write('\nLast Result:\n')
                f.write('Acc: %f, Spe: %f, Sen: %f, AUC: %f, Pre: %f, F1score: %f'
                        % (val_acc, spe, sen, auc, pre, f1score))

        avg_epoch_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Train/LR', optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)
        writer.add_scalar('Train/Acc', cm.diag().sum() / cm.sum(), global_step=epoch)
        writer.add_scalar('Train/Avg_epoch_loss', avg_epoch_loss, global_step=epoch)


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
    train_set = utils.get_dataset(args.data_path, args.csv_path, args.clinical_path, args.img_size_US, args.img_size_MM,
                                  mode='train')
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
    train_idx = np.where(data_split == 1)[0].tolist()
    test_idx = np.where(data_split == 2)[0].tolist()

    print("\n")
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(test_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, sampler=train_sampler,
                              num_workers=4)
    val_loader = DataLoader(test_set, batch_size=1, shuffle=False, sampler=val_sampler)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, sampler=test_sampler)
    train(args, train_loader, test_loader, val_loader)
