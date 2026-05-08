import torch
import torch.nn as nn
from torch.optim.adam import Adam
from torch.profiler import ProfilerActivity, profile, schedule
from torch.utils.data import DataLoader, Dataset


class SimpleDataset(Dataset):
    """Simple dataset that creates random data on CPU"""

    def __init__(self, num_samples=1000, input_size=512):
        self.num_samples = num_samples
        self.input_size = input_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Create data on CPU with sin operations to make loading take time
        x = torch.randn(self.input_size, device="cpu")

        # Apply sin operations multiple times to make it time-consuming
        for i in range(10):  # Multiple sin operations
            x = torch.sin(x)

        # Simple target (binary classification)
        y = torch.tensor(1 if torch.sum(x[:10]).item() > 0 else 0, device="cpu").float()

        return x, y


class SimpleNN(nn.Module):
    """Simple neural network for profiling"""

    def __init__(self, input_size=512, hidden_size=256, num_classes=2):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def main():
    worker_name = "nn_a2"

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dataset and dataloader - smaller dataset and batch size to see data loading impact
    dataset = SimpleDataset(num_samples=320, input_size=512)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)

    # Create model, loss, and optimizer
    model = SimpleNN(input_size=512, hidden_size=256, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # Profile the training process with schedule
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(
            wait=2,
            warmup=2,
            active=2,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            "trace_nn", worker_name=worker_name, use_gzip=True
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,  # Track floating point operations
    ) as prof:
        model.train()

        # Train for more batches to see the scheduled profiling in action
        for step, (data, target) in enumerate(dataloader):
            # if step >= 20:
            #     break

            # Transfer data to GPU
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True).long().squeeze()

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            output = model(data)

            # Compute loss
            loss = criterion(output, target)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Step the profiler
            prof.step()

    # Export memory timeline
    prof.export_memory_timeline(f"trace_nn/{worker_name}_training_memory_timeline.json")

    # Print profiling results
    print("\n=== PROFILING RESULTS BY CPU TIME ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

    print("\n=== PROFILING RESULTS BY CUDA TIME ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


if __name__ == "__main__":
    main()
