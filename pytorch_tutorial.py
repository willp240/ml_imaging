import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# download training data from open datasets
#SETUP = True
#if SETUP == True:
training_data = datasets.FashionMNIST(root      = "data",
                                      train     = True,
                                      download  = True,
                                      transform = ToTensor())
test_data     = datasets.FashionMNIST(root      = "data",
                                      train     = False,
                                      download  = True,
                                      transform = ToTensor())
batch_size = 64

# create data loaders
train_dataLoader = DataLoader(training_data, batch_size = batch_size)
test_dataLoader  = DataLoader(test_data, batch_size = batch_size)

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

#figure = plt.figure(figsize=(8, 8))
#cols, rows = 3, 3
#for i in range(1, cols * rows + 1):
#    sample_idx = torch.randint(len(training_data), size=(1,)).item()
#    img, label = training_data[sample_idx]
#    figure.add_subplot(rows, cols, i)
#    plt.title(labels_map[label])
#    plt.axis("off")
#    plt.imshow(img.squeeze(), cmap="gray")
#plt.show()


#for X, y in test_dataLoader:
#    print(f"Shape of X [N, C, H, W]: {X.shape}")
#    print(f"Shape of y: {y.shape} {y.dtype}")
#    break

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# define the model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)

print(model)

loss_fn   = nn.CrossEntropyLoss()
optimiser = torch.optim.SGD(model.parameters(), lr = 1e-3)

def train(dataLoader, model, loss_fn, optimiser):
    size = len(dataLoader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataLoader):
        X, y = X.to(device), y.to(device)

        # compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # back propagation
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"Loss: {loss:>7f} [{current:>5d}]/{size:>5d}")

def test(dataLoader, model, loss_fn):
    model.eval()
    size = len(dataLoader.dataset)
    num_batches = len(dataLoader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataLoader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {100*correct:>0.1f}%, Avg Loss: {test_loss:>8f} \n")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataLoader, model, loss_fn, optimiser)
    test(test_dataLoader, model, loss_fn)
print("Done!")

torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")

model_loaded = NeuralNetwork().to(device)
model_loaded.load_state_dict(torch.load("model.pth"))

# can now make predictions with this model
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x  = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f"Predicted: {predicted} | Actual: {actual}")
