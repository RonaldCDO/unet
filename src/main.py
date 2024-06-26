import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

from dataset import PlantSegmentationDataset
from unet import Unet
from evaluation_methods import iou, dice_coefficient, accuracy
import os
from dotenv import load_dotenv

torch.cuda.empty_cache()
 
load_dotenv()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

scaler = torch.cuda.amp.GradScaler()

if __name__ == "__main__":
    LEARNING_RATE=float(os.getenv('LEARNING_RATE'))
    BATCH_SIZE=int(os.getenv('BATCH_SIZE'))
    EPOCHS=int(os.getenv('EPOCHS'))
    DATA_PATH=os.getenv('DATA_PATH')
    MODEL_SAVE_PATH=os.getenv('MODEL_SAVE_PATH')

    train_losses = []
    train_iou_scores = []
    train_dice_scores = []
    train_accuracy_scores = []

    val_losses = []
    val_iou_scores = []
    val_dice_scores = []
    val_accuracy_scores = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = PlantSegmentationDataset(DATA_PATH)

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(train_dataset, [0.8, 0.2], generator=generator)

    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=BATCH_SIZE,
                                  shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True)
    model = Unet(in_channels=3, num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    criterion = nn.BCEWithLogitsLoss()
    accumulation_steps = 4

    start = time.time()

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0
        train_running_iou = 0
        train_running_dice = 0
        train_running_accuracy = 0
        for idx, img_mask in enumerate(tqdm(train_dataloader)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            with torch.cuda.amp.autocast_mode.autocast():
                y_pred = model(img)
                loss = criterion(y_pred, mask)
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (idx + 1) % accumulation_steps == 0 or (idx + 1 == len(train_dataloader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_running_loss += loss.item() * accumulation_steps

            with torch.no_grad():
                # Calculating IoU, Dice Coefficient and Accuracy
                y_pred = torch.sigmoid(y_pred)
                batch_iou = iou(y_pred, mask)
                batch_dice = dice_coefficient(y_pred, mask)
                batch_accuracy = accuracy(y_pred, mask)

                train_running_iou += batch_iou
                train_running_dice += batch_dice
                train_running_accuracy += batch_accuracy

        train_loss = train_running_loss / len(train_dataloader)
        train_iou = train_running_iou / len(train_dataloader)
        train_dice = train_running_dice / len(train_dataloader)
        train_accuracy = train_running_accuracy / len(train_dataloader)

        train_losses.append(train_loss)
        train_iou_scores.append(train_iou)
        train_dice_scores.append(train_dice)
        train_accuracy_scores.append(train_accuracy)

        model.eval()
        val_running_loss = 0
        val_running_iou = 0
        val_running_dice = 0
        val_running_accuracy = 0
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)

                y_pred = model(img)
                loss = criterion(y_pred, mask)
                val_running_loss += loss.item()

                # Calculating IoU, Dice Coefficient and Accuracy
                y_pred = torch.sigmoid(y_pred)
                batch_iou = iou(y_pred, mask)
                batch_dice = dice_coefficient(y_pred, mask)
                batch_accuracy = accuracy(y_pred, mask)

                val_running_iou += batch_iou
                val_running_dice += batch_dice
                val_running_accuracy += batch_accuracy

        val_loss = val_running_loss / len(val_dataloader)
        val_iou = val_running_iou / len(val_dataloader)
        val_dice = val_running_dice / len(val_dataloader)
        val_accuracy = val_running_accuracy / len(val_dataloader)

        val_losses.append(val_loss)
        val_iou_scores.append(val_iou)
        val_dice_scores.append(val_dice)
        val_accuracy_scores.append(val_accuracy)


        print("-"*30)
        print(f"EPOCH {epoch+1}/{EPOCHS}: Loss {train_loss:.4f}, IoU {train_iou:.4f}, Dice {train_dice:.4f}, Accuracy {train_accuracy:.4f}")
        print("-"*30)
        if (epoch+1) % 10 == 0:
            model_save_dir = f'{MODEL_SAVE_PATH}/{epoch+1}_epochs'
            os.makedirs(model_save_dir, exist_ok=True)

            torch.save(model.state_dict(), f'{model_save_dir}/unet.pt')
            fig, ax = plt.subplots()
            plt.clf()
            ax.plot(range(1, epoch+2), train_losses, label='Training Loss')
            ax.plot(range(1, epoch+2), val_losses, label='Validation Loss')
            ax.set_ylabel('Loss/Error')
            ax.set_xlabel('Epoch')
            ax.set_title('Loss x Epochs')

            ax.legend(loc='upper right', shadow=True)
            plt.savefig(f'{model_save_dir}/loss.png')

            actual = time.time()
            print(f"Training time: {actual-start:.2f}s")

            with open(f'{model_save_dir}/training_time.txt', 'w') as f:
                f.write('Epochs: ' + str(EPOCHS) + '\n')
                f.write(f"Training time: {actual-start:.2f}s")

            plt.clf()
            fig, ax = plt.subplots()
            ax.plot(range(1, epoch+2), train_iou_scores, label='Training IoU')
            ax.plot(range(1, epoch+2), val_iou_scores, label='Validation IoU')
            ax.set_ylabel('IoU')
            ax.set_xlabel('Epoch')
            ax.set_title('IoU x Epochs')

            ax.legend(loc='upper left', shadow=True)
            plt.savefig(f'{model_save_dir}/iou.png')

            plt.clf()
            fig, ax = plt.subplots()
            ax.plot(range(1, epoch+2), train_dice_scores, label='Training Dice')
            ax.plot(range(1, epoch+2), val_dice_scores, label='Validation Dice')
            ax.set_ylabel('Dice')
            ax.set_xlabel('Epoch')
            ax.set_title('Dice x Epochs')

            ax.legend(loc='upper left', shadow=True)
            plt.savefig(f'{model_save_dir}/dice.png')

            plt.clf()
            fig, ax = plt.subplots()
            ax.plot(range(1, epoch+2), train_accuracy_scores, label='Training Accuracy')
            ax.plot(range(1, epoch+2), val_accuracy_scores, label='Validation Accuracy')
            ax.set_ylabel('Accuracy')
            ax.set_xlabel('Epoch')
            ax.set_title('Accuracy x Epochs')

            ax.legend(loc='upper left', shadow=True)
            plt.savefig(f'{model_save_dir}/accuracy.png')

    end = time.time()
    print(f"Total training time: {end-start:.2f}s")
