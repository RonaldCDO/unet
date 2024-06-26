import os
import torch
import matplotlib.pyplot as plt
import sys
from torchvision import transforms
from PIL import Image

from dataset import PlantSegmentationDataset
from unet import Unet
from dotenv import load_dotenv
import numpy as np
 
load_dotenv()

def pred_show_image_grid(data_path, model_pt, device, inference_path):
    model = Unet(in_channels=3, num_classes=1).to(device)
    model.load_state_dict(torch.load(model_pt, map_location=torch.device(device)))
    image_dataset = PlantSegmentationDataset(data_path, test=True)
    images = []
    orig_masks = []
    pred_masks = []
    overlays = []

    for img, orig_mask in image_dataset:
        img = img.float().to(device)
        img = img.unsqueeze(0)

        with torch.no_grad():
            pred_mask = model(img)
            pred_mask = torch.sigmoid(pred_mask)

        pred_mask = pred_mask.squeeze(0).cpu().detach()
        pred_mask = pred_mask.permute(1, 2, 0).squeeze()

        pred_mask = (pred_mask > 0.5).float()

        orig_mask = orig_mask.cpu().detach()
        orig_mask = orig_mask.permute(1, 2, 0).squeeze()

        img_cpu = img.squeeze(0).cpu().numpy().transpose(1,2,0)

        color_mask = np.zeros_like(img_cpu, dtype=np.uint8)
        color_mask[:,:,0] = (pred_mask.numpy() * 255).astype(np.uint8)

        alpha = 0.3
        overlay = img_cpu.copy()
        overlay[color_mask[:, :, 0] > 0] = (
            overlay[color_mask[:, :, 0] > 0] * (1 - alpha) + color_mask[color_mask[:, :, 0] > 0] * alpha
        )


        images.append(img_cpu)
        orig_masks.append(orig_mask)
        pred_masks.append(pred_mask)
        overlays.append(overlay)

    plot_images = images + orig_masks + pred_masks + overlays
    titles = ['Original Image', 'Original Mask', 'Predicted Mask', 'Segmentation Overlay']

    _, axs = plt.subplots(4, len(image_dataset), figsize=(20,20))
    for i, ax in enumerate(axs.flat):
        img_idx = i%len(image_dataset)
        img_type_idx = i//len(image_dataset)
        if img_type_idx == 0:
            ax.imshow(plot_images[img_idx], cmap='gray')
        elif img_type_idx == 1:
            ax.imshow(plot_images[len(image_dataset) + img_idx], cmap='gray')
        elif img_type_idx == 2:
            ax.imshow(plot_images[2*len(image_dataset) + img_idx], cmap='gray')
        else:
            ax.imshow(plot_images[3*len(image_dataset) + img_idx])
        ax.set_title(titles[img_type_idx])
        ax.axis('off')
    plt.savefig(f'{inference_path}group_image_inference')
    plt.show()
 
def single_image_inference(img_pt, model_pt, device, inference_path):
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
    pred_mask = pred_mask.permute(1, 2, 0).squeeze()

    # threshold verification
    # plt.figure()
    # plt.imshow(pred_mask, cmap='gray')
    # plt.title('Model Output Before Thresholding')
    # plt.colorbar()
    # plt.show()

    pred_mask = (pred_mask > 0.1).float()

    img_cpu = img.squeeze(0).cpu().numpy().transpose(1,2,0)

    color_mask = np.zeros_like(img_cpu, dtype=np.uint8)
    color_mask[:,:,0] = (pred_mask.numpy() * 255).astype(np.uint8)

    alpha = 0.3
    overlay = img_cpu.copy()
    overlay[color_mask[:, :, 0] > 0] = (
        overlay[color_mask[:, :, 0] > 0] * (1 - alpha) + color_mask[color_mask[:, :, 0] > 0] * alpha
    )

    fig = plt.figure()
    for i in range(1, 4):
        fig.add_subplot(1,3, i)
        if i == 1:
            plt.imshow(img_cpu, cmap='gray')
            plt.title('Original Image')
        elif i == 2:
            plt.imshow(pred_mask, cmap='gray')
            plt.title('Predicted Mask')
        else:
            plt.imshow(overlay)
            plt.title('Segmentation Overlay')
    plt.savefig(f'{inference_path}single_image_inference.png')
    plt.show()

if __name__ == '__main__':
    SINGLE_IMG_PATH=os.getenv('SINGLE_IMG_PATH')
    DATA_PATH=os.getenv('DATA_PATH')
    MODEL_PATH=os.getenv('MODEL_PATH')
    INFERENCE_PATH=os.getenv('INFERENCE_PATH')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if sys.argv[1] == 'single':
        single_image_inference(SINGLE_IMG_PATH, MODEL_PATH, device, INFERENCE_PATH)
    if sys.argv[1] == 'group':
        pred_show_image_grid(DATA_PATH, MODEL_PATH, device, INFERENCE_PATH)



