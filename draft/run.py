import os
import torch
import torchvision.transforms as transforms
from torchvision.models.segmentation import fcn_resnet101
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Define custom dataset class
class CustomDataset(Dataset):
    def __init__(self, image_folder, mask_folder, transform=None):
        self.image_folder = image_folder
        self.mask_folder = mask_folder
        self.transform = transform

        # Get the list of filenames in the image folder
        self.filenames = sorted(os.listdir(image_folder))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        # Get the filename for the current index
        filename = self.filenames[idx]

        # Construct the file paths for the image and mask
        img_path = os.path.join(self.image_folder, filename)
        mask_path = os.path.join(self.mask_folder, filename)

        # Load the image and mask
        image = Image.open(img_path)
        mask = Image.open(mask_path)

        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Convert the mask to a binary mask where 1 indicates the presence of solar panels
        mask = torch.tensor(np.array(mask) == 255, dtype=torch.float32)

        return image, mask

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to a suitable input size
    transforms.ToTensor(),
])

# Define paths for the new folders
image_folder = './data/load/img/'
mask_folder = './data/load/masks/'

# Create custom dataset
dataset = CustomDataset(image_folder, mask_folder, transform)

# Split the dataset into training and validation sets
train_dataset, val_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Load pre-trained FCN-ResNet101 model
model = fcn_resnet101(pretrained=True)

# Modify the last layer to match the number of classes (assuming binary segmentation)
model.classifier[4] = torch.nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = torch.nn.BCEWithLogitsLoss()  # Binary Cross Entropy Loss for binary segmentation
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for images, masks in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs['out'][:, 0, :, :], masks)
        loss.backward()
        optimizer.step()

# Validation loop
model.eval()
total_val_loss = 0.0
with torch.no_grad():
    for images, masks in tqdm(val_loader, desc='Validation'):
        images, masks = images.to(device), masks.to(device)
        outputs = model(images)
        loss = criterion(outputs['out'][:, 0, :, :], masks)
        total_val_loss += loss.item()

average_val_loss = total_val_loss / len(val_loader)
print(f'Average Validation Loss: {average_val_loss}')

# Save the trained model
torch.save(model.state_dict(), 'fcn_resnet101_segmentation_model.pth')

