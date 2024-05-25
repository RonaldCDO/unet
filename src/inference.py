import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

from dataset import PlantSegmentationDataset
from unet import Unet
from dotenv import load_dotenv
 
load_dotenv()

def pred_show_image_grid(data_path, model_pt, device):
    model = Unet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pt, map_location=torch.device(device)))
    image_dataset = PlantSegmentationDataset(data_path, test=True)
    images = []
    orig_masks = []
    pred_masks = []

    for img, orig_mask in image_dataset:
        img = img.float().to(device)
        img = img.unsqueeze(0)

        with torch.no_grad():
            pred_mask = model(img)
            pred_mask = torch.sigmoid(pred_mask)

        img = img.squeeze(0).cpu().detach()
        img = img.permute(1,2,0)

        pred_mask = pred_mask.squeeze(0).cpu().detach()
        pred_mask = pred_mask.permute(1, 2, 0)

        pred_mask = (pred_mask > 0.1).float()

        orig_mask = orig_mask.cpu().detach()
        orig_mask = orig_mask.permute(1, 2, 0)


        images.append(img)
        orig_masks.append(orig_mask)
        pred_masks.append(pred_mask)

    images.extend(orig_masks)
    images.extend(pred_masks)
    fig = plt.figure()
    for i in range(1, 3*len(image_dataset)+1):
        fig.add_subplot(3, len(image_dataset), i)
        plt.imshow(images[i-1], cmap='gray')
    plt.show()

def single_image_inference(img_pt, model_pt, device):
    model= Unet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pt, map_location=torch.device(device)))
    model.eval()

    transform  = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()])

    img = transform(Image.open(img_pt)).float().to(device)
    img = img.unsqueeze(0)

    with torch.no_grad():
        pred_mask = model(img)
        pred_mask = torch.sigmoid(pred_mask)


    # debug
    # print("Raw model output (min, max):", pred_mask.min().item(), pred_mask.max().item())


    pred_mask = pred_mask.squeeze(0).cpu().detach()
    pred_mask = pred_mask.permute(1, 2, 0)

    plt.figure()
    plt.imshow(pred_mask, cmap='gray')
    plt.title('Model Output Before Thresholding')
    plt.colorbar()
    plt.show()

    pred_mask = (pred_mask > 0.1).float()

    img_cpu = img.squeeze(0).cpu().numpy().transpose(1,2,0)

    fig = plt.figure()
    for i in range(1, 3):
        fig.add_subplot(1,2, i)
        if i == 1:
            plt.imshow(img_cpu, cmap='gray')
        else:
            plt.imshow(pred_mask, cmap='gray')
    plt.show()

if __name__ == '__main__':
    SINGLE_IMG_PATH=os.getenv('SINGLE_IMG_PATH')
    DATA_PATH=os.getenv('DATA_PATH')
    MODEL_PATH=os.getenv('MODEL_PATH')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    pred_show_image_grid(DATA_PATH, MODEL_PATH, device)
    single_image_inference(SINGLE_IMG_PATH, MODEL_PATH, device)



