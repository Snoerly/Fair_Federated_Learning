#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torchvision import datasets, transforms
from sampling import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling import cifar_iid, cifar_noniid, tabular_iid
from sampling import tabular_noniid_label_skew, tabular_noniid_feature_skew
from tabular_datasets import (
    load_adult_dataset,
    load_bank_dataset,
    load_census_income_kdd_dataset,
    load_communities_crime_dataset,
)


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        apply_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                user_groups = cifar_noniid(train_dataset, args.num_users)

    elif args.dataset in ['mnist', 'fmnist']:
        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
        else:
            data_dir = '../data/fmnist/'

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    elif args.dataset == 'adult':
        # Tabellarischer Adult-Datensatz aus CSV
        data_dir = '../data/adult/'
        train_dataset, test_dataset, num_classes = load_adult_dataset(
            data_dir, sensitive_attr=getattr(args, 'sensitive_attr', 'sex')
        )

        # Falls num_classes nicht zu den Argumenten passt, überschreiben
        if hasattr(args, 'num_classes'):
            args.num_classes = num_classes

        # Aufteilung der Zeilen auf Clients
        if args.iid:
            # zufällige (IID) Splits
            user_groups = tabular_iid(train_dataset, args.num_users)
        else:
            # Non-IID Splits: label- oder feature-skew
            split_type = getattr(args, 'tabular_noniid', 'label-skew')
            if split_type == 'feature-skew':
                user_groups = tabular_noniid_feature_skew(train_dataset, args.num_users)
            else:
                user_groups = tabular_noniid_label_skew(train_dataset, args.num_users)

    elif args.dataset == 'bank':
        # Tabellarischer Bank-Datensatz aus CSV
        data_dir = '../data/bank/'
        train_dataset, test_dataset, num_classes = load_bank_dataset(
            data_dir, sensitive_attr=getattr(args, 'sensitive_attr', 'age')
        )

        if hasattr(args, 'num_classes'):
            args.num_classes = num_classes

        if args.iid:
            user_groups = tabular_iid(train_dataset, args.num_users)
        else:
            split_type = getattr(args, 'tabular_noniid', 'label-skew')
            if split_type == 'feature-skew':
                user_groups = tabular_noniid_feature_skew(train_dataset, args.num_users)
            else:
                user_groups = tabular_noniid_label_skew(train_dataset, args.num_users)

    elif args.dataset == 'census_income_kdd':
        # Tabellarischer Census Income KDD-Datensatz aus CSV
        data_dir = '../data/census_income_kdd/'
        train_dataset, test_dataset, num_classes = load_census_income_kdd_dataset(
            data_dir, sensitive_attr=getattr(args, 'sensitive_attr', 'ASEX')
        )

        if hasattr(args, 'num_classes'):
            args.num_classes = num_classes

        if args.iid:
            user_groups = tabular_iid(train_dataset, args.num_users)
        else:
            split_type = getattr(args, 'tabular_noniid', 'label-skew')
            if split_type == 'feature-skew':
                user_groups = tabular_noniid_feature_skew(train_dataset, args.num_users)
            else:
                user_groups = tabular_noniid_label_skew(train_dataset, args.num_users)

    elif args.dataset == 'communities_crime':
        # Tabellarischer Communities and Crime-Datensatz aus CSV
        data_dir = '../data/communities_crime/'
        train_dataset, test_dataset, num_classes = load_communities_crime_dataset(
            data_dir, sensitive_attr=getattr(args, 'sensitive_attr', 'racepctblack')
        )

        if hasattr(args, 'num_classes'):
            args.num_classes = num_classes

        if args.iid:
            user_groups = tabular_iid(train_dataset, args.num_users)
        else:
            split_type = getattr(args, 'tabular_noniid', 'label-skew')
            if split_type == 'feature-skew':
                user_groups = tabular_noniid_feature_skew(train_dataset, args.num_users)
            else:
                user_groups = tabular_noniid_label_skew(train_dataset, args.num_users)
            
    return train_dataset, test_dataset, user_groups


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
