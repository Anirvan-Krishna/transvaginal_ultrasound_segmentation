import torch
from byol_pytorch import BYOL
from torchvision import models
from config import Dataset
from tqdm import tqdm

n_epochs = 100
lr = 3e-4
batch_size = 32
device = 'cuda' if torch.cuda.is_available() else 'cpu'

resnet = models.resnet50(pretrained=True)


learner = BYOL(
    resnet,
    image_size=256,
    hidden_layer='avgpool'
).to(device)

opt = torch.optim.Adam(learner.parameters(), lr=lr)

def prepare_unlabelled_images():
    unlabelled_img_path = "train\\unlabeled_data\\images"
    unlabelled_dataset = Dataset(image_path=unlabelled_img_path)
    unlabelled_loader = torch.utils.data.DataLoader(unlabelled_dataset, 
                                                    batch_size=batch_size, 
                                                    shuffle=True,
                                                    num_workers=2,
                                                    pin_memory=True if device == 'cuda' else False)

    return unlabelled_loader

dataloader = prepare_unlabelled_images()

# for logging
epoch_losses = []

for epoch in range(n_epochs):
    total_loss = 0
    num_batches = 0
    
    for _, images in enumerate(tqdm(dataloader)):
        images = images.to(device)
        
        loss = learner(images)
        total_loss += loss.item()
        num_batches += 1
        
        opt.zero_grad()
        loss.backward()
        opt.step()
        learner.update_moving_average()
    
    avg_loss = total_loss / num_batches
    epoch_losses.append(avg_loss)
    print(f'Epoch [{epoch+1}/{n_epochs}], Average Loss: {avg_loss:.4f}')

# save your improved network
torch.save(resnet.state_dict(), './byol_encoder.pt')
