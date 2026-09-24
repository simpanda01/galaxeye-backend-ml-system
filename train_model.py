import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
from torch.utils.data import random_split, DataLoader


DATA_DIR = "data/candidate_tiles"
MODEL_PATH = "satellite_model.pth"

BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

device = torch.device("cpu")

print(f"Using device: {device}")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform
)

print("Classes:", dataset.classes)
print("Total images:", len(dataset))


train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)


for parameter in model.parameters():
    parameter.requires_grad = False


model.fc = nn.Linear(
    model.fc.in_features,
    len(dataset.classes)
)

model = model.to(device)


criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_accuracy = 100 * correct / total


    model.eval()

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, 1)

            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = 100 * val_correct / val_total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {running_loss / len(train_loader):.4f} | "
        f"Train Accuracy: {train_accuracy:.2f}% | "
        f"Validation Accuracy: {val_accuracy:.2f}%"
    )


torch.save(
    {
        "model_state_dict": model.state_dict(),
        "classes": dataset.classes
    },
    MODEL_PATH
)

print()
print(f"Model saved to: {MODEL_PATH}")