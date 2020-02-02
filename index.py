import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
from torchvision.datasets import CIFAR10


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x, **kwargs):
        batch_size = x.shape[0]
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return x


def test(model, data_loader):
    print("Test started...")
    correct = 0
    total = 0
    with torch.no_grad():
        for data, labels in data_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))


def train(model, data_loader):
    optimizer = optim.Adam(model.parameters())
    criterion = nn.NLLLoss()

    for data, labels in tqdm.tqdm(data_loader):
        optimizer.zero_grad()
        loss = criterion(model(data), labels)
        loss.backward()        
        optimizer.step()


if __name__ == '__main__':
    num_processes = 4
    model = Model()
    model.share_memory()

    dataset = CIFAR10(
                "data",
                train=True,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
            )

    testset = CIFAR10(
        "data",
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    )

    processes = []
    for rank in range(num_processes):
        data_loader = DataLoader(
            dataset=dataset,
            sampler=DistributedSampler(
                dataset=dataset,
                num_replicas=num_processes,
                rank=rank
            ),
            batch_size=32
        )
        p = mp.Process(target=train, args=(model, data_loader))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    test(model, DataLoader(
        dataset=testset,
        batch_size=1000
    ))
