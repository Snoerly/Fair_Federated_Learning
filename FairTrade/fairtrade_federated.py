from __future__ import annotations

import argparse
import copy
import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent

# Re-use the GIFAIR tabular pipeline and fairness metrics from a local copy to
# avoid reaching outside this repository.
from GIFAIR.utils import get_dataset, average_weights
from GIFAIR.fairness_metrics import compute_fairness_metrics
from constraint import DemographicParityLoss, AverageTreatmentEffectLoss


@dataclass
class Args:
    # Federated config
    epochs: int
    num_users: int
    frac: float
    local_ep: int
    local_bs: int
    lr: float
    optimizer: str
    dataset: str
    num_classes: int
    iid: int
    unequal: int
    tabular_noniid: str
    sensitive_attr: str
    seed: int
    model: str
    # FairTrade-specific
    fairness_notion: str
    fairness_lambda: float
    device: str


def args_parser() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_users', type=int, default=3)
    parser.add_argument('--frac', type=float, default=1.0)
    parser.add_argument('--local_ep', type=int, default=5)
    parser.add_argument('--local_bs', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'])
    parser.add_argument('--dataset', type=str, default='adult', choices=['adult', 'bank', 'census_income_kdd', 'communities_crime'])
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--iid', type=int, default=1)
    parser.add_argument('--unequal', type=int, default=0)
    parser.add_argument('--tabular_noniid', type=str, default='label-skew', choices=['label-skew', 'feature-skew'])
    parser.add_argument('--sensitive_attr', type=str, default='sex')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--fairness_notion', type=str, default='stat_parity', choices=['stat_parity', 'ate'])
    parser.add_argument('--fairness_lambda', type=float, default=1.0)
    parser.add_argument('--gpu', action='store_true', help='Use CUDA if available')

    ns = parser.parse_args()
    device = 'cuda' if ns.gpu and torch.cuda.is_available() else 'cpu'
    return Args(
        epochs=ns.epochs,
        num_users=ns.num_users,
        frac=ns.frac,
        local_ep=ns.local_ep,
        local_bs=ns.local_bs,
        lr=ns.lr,
        optimizer=ns.optimizer,
        dataset=ns.dataset,
        num_classes=ns.num_classes,
        iid=ns.iid,
        unequal=ns.unequal,
        tabular_noniid=ns.tabular_noniid,
        sensitive_attr=ns.sensitive_attr,
        seed=ns.seed,
        model=ns.model,
        fairness_notion=ns.fairness_notion,
        fairness_lambda=ns.fairness_lambda,
        device=device,
    )


class DatasetSplit(Dataset):
    """Subset wrapper that also exposes sensitive groups if present."""

    def __init__(self, dataset: Dataset, idxs: Sequence[int]):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self) -> int:
        return len(self.idxs)

    def __getitem__(self, item: int):
        idx = self.idxs[item]
        x, y = self.dataset[idx]
        if hasattr(self.dataset, 'groups') and self.dataset.groups is not None:
            g = self.dataset.groups[idx]
            return x, y, g
        return x, y


class TabularMLP(nn.Module):
    def __init__(self, dim_in: int, dim_hidden: int, dim_out: int):
        super().__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.layer_input(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer_hidden(x)
        return x  # logits


class LocalUpdate:
    def __init__(self, args: Args, dataset: Dataset, idxs: Sequence[int]):
        self.args = args
        self.trainloader, self.validloader, self.testloader = self.train_val_test(dataset, idxs)
        self.device = args.device
        self.criterion = nn.BCEWithLogitsLoss()
        # Select fairness loss
        if args.fairness_notion == 'stat_parity':
            self.fairness_loss = DemographicParityLoss(alpha=1.0)
            self.needs_y = False
        else:
            self.fairness_loss = AverageTreatmentEffectLoss(alpha=1.0)
            self.needs_y = True

    def train_val_test(self, dataset: Dataset, idxs: Sequence[int]) -> Tuple[DataLoader, DataLoader, DataLoader]:
        idxs = list(idxs)
        split_1 = int(0.8 * len(idxs))
        split_2 = int(0.9 * len(idxs))
        idxs_train = idxs[:split_1]
        idxs_val = idxs[split_1:split_2]
        idxs_test = idxs[split_2:]

        def _loader(sub_idxs: Sequence[int], batch_size: int, shuffle: bool) -> DataLoader:
            return DataLoader(DatasetSplit(dataset, sub_idxs), batch_size=batch_size, shuffle=shuffle)

        return (
            _loader(idxs_train, batch_size=max(1, self.args.local_bs), shuffle=True),
            _loader(idxs_val, batch_size=max(1, len(idxs_val) // 2 or 1), shuffle=False),
            _loader(idxs_test, batch_size=max(1, len(idxs_test) // 2 or 1), shuffle=False),
        )

    def update_weights(self, model: nn.Module, global_round: int) -> Tuple[dict, float]:
        model.train()
        optimizer = self._make_optimizer(model)
        epoch_loss: List[float] = []

        for _ in range(self.args.local_ep):
            batch_loss: List[float] = []
            for batch in self.trainloader:
                if len(batch) == 3:
                    images, labels, groups = batch
                    groups = groups.to(self.device)
                else:
                    images, labels = batch
                    groups = None
                images = images.to(self.device)
                labels = labels.to(self.device).float()

                optimizer.zero_grad()
                logits = model(images).view(-1)
                primary_loss = self.criterion(logits, labels)
                fairness_term = torch.tensor(0.0, device=self.device)
                if groups is not None:
                    fairness_term = self.fairness_loss(
                        images, logits, groups,
                        y=labels if self.needs_y else None,
                    )
                loss = primary_loss + self.args.fairness_lambda * fairness_term
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return model.state_dict(), float(sum(epoch_loss) / len(epoch_loss))

    def inference(self, model: nn.Module) -> Tuple[float, float]:
        model.eval()
        loss, correct, total = 0.0, 0.0, 0.0
        criterion = self.criterion
        with torch.no_grad():
            for batch in self.testloader:
                if len(batch) == 3:
                    images, labels, _groups = batch
                else:
                    images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device).float()
                logits = model(images).view(-1)
                batch_loss = criterion(logits, labels)
                loss += batch_loss.item()
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.numel()
        return float(correct / total), float(loss)

    def _make_optimizer(self, model: nn.Module):
        if self.args.optimizer == 'sgd':
            return torch.optim.SGD(model.parameters(), lr=self.args.lr, momentum=0.5)
        if self.args.optimizer == 'adam':
            return torch.optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)
        raise ValueError(f"Unsupported optimizer: {self.args.optimizer}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(args: Args, train_dataset: Dataset) -> nn.Module:
    sample_x, _ = train_dataset[0]
    dim_in = int(np.prod(sample_x.shape))
    # Use a single logit for binary classification to match BCEWithLogitsLoss
    model = TabularMLP(dim_in=dim_in, dim_hidden=64, dim_out=1)
    return model.to(args.device)


def run() -> None:
    args = args_parser()
    set_seed(args.seed)

    # Load datasets and client partitions via GIFAIR pipeline
    train_dataset, test_dataset, user_groups = get_dataset(args)
    global_model = build_model(args, train_dataset)
    global_weights = global_model.state_dict()

    results_dir = ROOT / 'save' / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_name = results_dir / (
        f"fairtrade_{args.dataset}_{args.model}_users[{args.num_users}]_iid[{args.iid}]_C[{args.frac}]_"
        f"E[{args.epochs}]_localE[{args.local_ep}]_B[{args.local_bs}]_"
        f"split[{args.tabular_noniid}]_sens[{args.sensitive_attr}]_seed[{args.seed}].csv"
    )
    headers = [
        'round', 'dataset', 'model', 'num_users', 'frac', 'iid',
        'tabular_noniid', 'epochs', 'local_ep', 'local_bs', 'seed',
        'train_loss', 'train_accuracy', 'test_accuracy',
        'eop_gap', 'di_ratio', 'tpr_priv', 'tpr_unpriv',
        'p_pos_unpriv', 'p_pos_priv', 'sensitive_attr',
    ]
    write_header = not csv_name.exists()
    with csv_name.open('a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(headers)

    logger_loss: List[float] = []
    logger_accuracy: List[float] = []

    for epoch in range(args.epochs):
        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        selected_users = np.random.choice(range(args.num_users), m, replace=False)

        local_weights: List[dict] = []
        local_losses: List[float] = []

        for user in selected_users:
            local = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[user])
            local_model = copy.deepcopy(global_model)
            w, loss = local.update_weights(model=local_model, global_round=epoch)
            local_weights.append(w)
            local_losses.append(loss)

        global_weights = average_weights(local_weights)
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        logger_loss.append(loss_avg)

        # Train accuracy over users
        acc_list: List[float] = []
        for user in range(args.num_users):
            local = LocalUpdate(args=args, dataset=train_dataset, idxs=user_groups[user])
            acc, _ = local.inference(model=global_model)
            acc_list.append(acc)
        train_acc = sum(acc_list) / len(acc_list)
        logger_accuracy.append(train_acc)

        # Evaluation on test set
        test_acc, _ = test_inference(args, global_model, test_dataset)

        # Fairness metrics
        fairness_results = None
        if hasattr(test_dataset, 'groups') and test_dataset.groups is not None:
            global_model.eval()
            y_true, y_pred = [], []
            loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(args.device)
                    y = y.to(args.device)
                    logits = global_model(x).view(-1)
                    preds = (torch.sigmoid(logits) > 0.5).long()
                    y_true.extend(y.cpu().numpy())
                    y_pred.extend(preds.cpu().numpy())
            groups = test_dataset.groups.cpu().numpy()
            fairness_results = compute_fairness_metrics(y_true, y_pred, groups)

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
            args.tabular_noniid,
            args.epochs,
            args.local_ep,
            args.local_bs,
            args.seed,
            float(logger_loss[-1]) if logger_loss else '',
            float(logger_accuracy[-1]) if logger_accuracy else '',
            float(test_acc),
            fr['eop_gap'],
            fr['di_ratio'],
            fr['tpr_priv'],
            fr['tpr_unpriv'],
            fr['p_pos_unpriv'],
            fr['p_pos_priv'],
            args.sensitive_attr,
        ]

        with csv_name.open('a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)

    print(f"Finished training. Results written to {csv_name}")


def test_inference(args: Args, model: nn.Module, test_dataset: Dataset) -> Tuple[float, float]:
    model.eval()
    loss, correct, total = 0.0, 0.0, 0.0
    criterion = nn.BCEWithLogitsLoss()
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    with torch.no_grad():
        for batch in testloader:
            if len(batch) == 3:
                images, labels, _groups = batch
            else:
                images, labels = batch
            images = images.to(args.device)
            labels = labels.to(args.device).float()
            logits = model(images).view(-1)
            batch_loss = criterion(logits, labels)
            loss += batch_loss.item()
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()
    return float(correct / total), float(loss)


if __name__ == '__main__':
    run()
