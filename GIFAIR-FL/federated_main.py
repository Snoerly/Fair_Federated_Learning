#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import random
import os
import copy
import random
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import csv

from options import args_parser
from update import LocalUpdate, test_inference, calculate
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details
from fairness_metrics import compute_fairness_metrics


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    #if args.gpu_id:
    #    torch.cuda.set_device(args.gpu_id)
    #device = 'cuda' if args.gpu_id else 'cpu'
    device = 'cpu'

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            global_model = CNNCifar(args=args)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0

    initial_loss = random.sample(range(args.num_users), args.num_users)
    users_pool = np.random.choice(range(args.num_users), args.num_users, replace=False)
    users_pool.sort()
    d_group = args.num_users
    r_k_values = []
    for i in range(args.num_users) :
        r_k_values.append(d_group - (2*(i+1)-1))
    loss_dic = {users_pool[i]: initial_loss[i] for i in range(len(users_pool))}
    loss_dic = dict(sorted(loss_dic.items(), key=lambda item: item[1], reverse=True))
    r_k_dic = {list(loss_dic.keys())[i]: r_k_values[i] for i in range(len(list(loss_dic.keys())))}


    weight_coefficient_p = []
    for idx in users_pool:
        local_model = calculate(args=args, dataset=train_dataset,
                                  idxs=user_groups[idx], logger=logger, r_k=r_k_dic[idx], lambda_reg=0.001, p = 0)
        a, b, d, c = local_model.train_val_test(dataset=train_dataset,
                                      idxs=user_groups[idx])
        weight_coefficient_p.append(c)

    total_size = sum(weight_coefficient_p)
    weight_coefficient_p = [number / total_size for number in weight_coefficient_p]

    new_reg = [number / (d_group - 1) for number in weight_coefficient_p]
    lambda_reg = min(new_reg)/2

    # Prepare per-round CSV logging (one row per global round).
    # Encode möglichst viele Konfigurationsparameter im Dateinamen,
    # damit verschiedene Läufe nicht kollidieren.
    csv_name = (
    "save/results/gfl_{}_users[{}]_iid[{}]_C[{}]_E[{}]_localE[{}]_B[{}]_split[{}]_sens[{}]_seed[{}].csv"
    ).format(
        args.dataset,
        args.num_users,
        args.iid,
        args.frac,
        args.epochs,
        args.local_ep,
        args.local_bs,
        getattr(args, "tabular_noniid", ""),
        getattr(args, "sensitive_attr", ""),
        args.seed,
    )
    os.makedirs(os.path.dirname(csv_name), exist_ok=True)
    write_header = not os.path.exists(csv_name)
    if write_header:
        headers = [
            'round', 'dataset', 'model', 'num_users', 'frac', 'iid',
            'tabular_noniid', 'epochs', 'local_ep', 'local_bs', 'seed',
            'train_loss', 'train_accuracy', 'test_accuracy',
            'eop_gap', 'di_ratio', 'tpr_priv', 'tpr_unpriv',
            'p_pos_unpriv', 'p_pos_priv',
            'sensitive_attr',
        ]
        with open(csv_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

    for epoch in tqdm(range(args.epochs)):
        loss_pool = [] # collect loss from all devices
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        # rank losses
        loss_dic = {users_pool[i]: initial_loss[i] for i in range(len(users_pool))}
        loss_dic = dict(sorted(loss_dic.items(), key=lambda item: item[1], reverse=True))
        r_k_dic = {list(loss_dic.keys())[i]: r_k_values[i] for i in range(len(list(loss_dic.keys())))}

        # training
        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False, p = weight_coefficient_p)

        for idx in idxs_users:
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger, r_k = r_k_dic[idx], lambda_reg = lambda_reg, p = weight_coefficient_p[idx])
            w, loss = local_model.update_weights(
                model=copy.deepcopy(global_model), global_round=epoch, r_k = r_k_dic[idx], lambda_reg = lambda_reg, p = weight_coefficient_p[idx])
            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))
            initial_loss[idx] = copy.deepcopy(loss)



        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss = [], []
        global_model.eval()
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger, r_k = r_k_dic[idx], lambda_reg = lambda_reg, p = weight_coefficient_p[idx])
            acc, loss = local_model.inference(model=global_model)
            list_acc.append(acc)
            list_loss.append(loss)
        train_accuracy.append(sum(list_acc)/len(list_acc))

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))

        # Per-round evaluation on test set
        test_acc, test_loss = test_inference(args, global_model, test_dataset)

        # Fairness evaluation (if groups available)
        fairness_results = None
        if hasattr(test_dataset, 'groups') and test_dataset.groups is not None:
            global_model.eval()
            y_true, y_pred = [], []

            loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(device)
                    y = y.to(device)
                    outputs = global_model(x)
                    _, preds = torch.max(outputs, 1)
                    y_true.extend(y.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())

            groups = test_dataset.groups.cpu().numpy()
            fairness_results = compute_fairness_metrics(y_true, y_pred, groups)

        # Write one CSV row for this global round
        fr = fairness_results or {
            'eop_gap': '', 'di_ratio': '', 'tpr_priv': '', 'tpr_unpriv': '',
            'p_pos_unpriv': '', 'p_pos_priv': '',
        }

        row = [
            epoch + 1,
            args.dataset,
            args.model,
            args.num_users,
            args.frac,
            args.iid,
            getattr(args, 'tabular_noniid', ''),
            args.epochs,
            args.local_ep,
            args.local_bs,
            args.seed,
            float(train_loss[-1]) if train_loss else '',
            float(train_accuracy[-1]) if train_accuracy else '',
            float(test_acc),
            fr['eop_gap'],
            fr['di_ratio'],
            fr['tpr_priv'],
            fr['tpr_unpriv'],
            fr['p_pos_unpriv'],
            fr['p_pos_priv'],
            getattr(args, 'sensitive_attr', ''),
        ]

        with open(csv_name, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)

    # Test inference after completion of training (final print only)
    test_acc, test_loss = test_inference(args, global_model, test_dataset)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy as pickle:
    file_name = 'save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    # ensure target directory exists
    os.makedirs(os.path.dirname(file_name), exist_ok=True)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # ensure plot directory exists
    plot_dir = os.path.join('..', 'save')
    os.makedirs(plot_dir, exist_ok=True)

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    loss_path = os.path.join(
        plot_dir,
        'fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_loss.png'.format(
            args.dataset, args.model, args.epochs, args.frac,
            args.iid, args.local_ep, args.local_bs,
        ),
    )
    plt.savefig(loss_path)

    # Plot Average Accuracy vs Communication rounds
    plt.figure()
    plt.title('Average Accuracy vs Communication rounds')
    plt.plot(range(len(train_accuracy)), train_accuracy, color='k')
    plt.ylabel('Average Accuracy')
    plt.xlabel('Communication Rounds')
    acc_path = os.path.join(
        plot_dir,
        'fed_{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}]_acc.png'.format(
            args.dataset, args.model, args.epochs, args.frac,
            args.iid, args.local_ep, args.local_bs,
        ),
    )
    plt.savefig(acc_path)
