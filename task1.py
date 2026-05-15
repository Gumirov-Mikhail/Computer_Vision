import statistics
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def prepare_data() -> TensorDataset:
    X = torch.randn(10000, 128)
    y = torch.randint(0, 2, (10000,))
    dataset = TensorDataset(X, y)
    return dataset


def train():
    # Уровень 2: врубаем num_workers для параллельной загрузки и pin_memory для ускорения передачи данных в память видеокарты
    dataloader = DataLoader(
        prepare_data(), 
        batch_size=256, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )

    model = nn.Sequential(
        nn.Linear(128, 512), nn.ReLU(),
        nn.Linear(512, 128), nn.ReLU(),
        nn.Linear(128, 2)
    ).cuda().train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    losses_history = []
    forward_times = []
    backward_times = []

    # Уровень 3: создаем эвенты для точного замера времени прямо на видеокарте
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for batch_idx, (data, target) in enumerate(dataloader):
        # Уровень 1: создаем шум сразу в видеопамяти, чтобы не копировать его лишний раз
        noise = torch.randn(data.shape, device='cuda')
        
        # Уровень 2: non_blocking=True позволяет передавать данные асинхронно, не тормозя основной поток
        data = data.to('cuda', non_blocking=True) + noise
        target = target.to('cuda', non_blocking=True)

        # Уровень 1: очищаем градиенты через set_to_none — это быстрее и бережет память от OOM
        optimizer.zero_grad(set_to_none=True)

        # Уровень 3: замеряем forward на видеокарте
        start_event.record()
        output = model(data)
        loss = criterion(output, target)
        end_event.record()
        torch.cuda.synchronize()
        forward_times.append(start_event.elapsed_time(end_event) / 1000.0)

        # Уровень 3: замеряем backward на видеокарте
        start_event.record()
        loss.backward()
        end_event.record()
        torch.cuda.synchronize()
        backward_times.append(start_event.elapsed_time(end_event) / 1000.0)
        
        optimizer.step()

        # Уровень 1: извлекаем значение через .item(), чтобы не копить графы вычислений в памяти
        losses_history.append(loss.item())
        
        # Уровень 2: убрали torch.cuda.empty_cache(), так как эта команда принудительно стопала асинхронную работу
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx} loss: {loss.item():.4f}")

    # Уровень 3: выводим среднее время работы железа
    print(f"Epoch finished, avg forward time is {statistics.mean(forward_times):.6f}s, "
          f"avg backward time is {statistics.mean(backward_times):.6f}s")


if __name__ == '__main__':
    train()
