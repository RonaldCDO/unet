import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from datetime import datetime
import pandas as pd

from dataset import PlantSegmentationDataset
from unet import Unet
from evaluation_methods import iou, dice_coefficient, accuracy
import os
from dotenv import load_dotenv

torch.cuda.empty_cache()

load_dotenv()

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

scaler = torch.cuda.amp.GradScaler()


def are_best_weights(
    loss: list,
    iou: list,
    dice: list,
    acc: list,
    loss_v: float,
    iou_v: float,
    dice_v: float,
    acc_v: float,
) -> bool:
    if len(loss) == 0:
        return False

    if (
        int(loss_v*100) < int(min(loss)*100)
        and int(iou_v*100) > int(max(iou)*100)
        and int(dice_v*100) > int(max(dice)*100)
        and (int(acc_v*100) >= int(max(acc)*100) or int(acc_v*100) > 900)
    ):
        return True
    return False


def generate_graph(
    train_list: list,
    val_list: list,
    epoch: int,
    x_label: str,
    y_label: str,
    metric: str,
    save_file_path: str,
) -> None:
    fig, ax = plt.subplots()
    ax.plot(range(1, epoch + 2), train_list, label=f"Training {metric}")
    ax.plot(range(1, epoch + 2), val_list, label=f"Validation {metric}")
    ax.set_ylabel(x_label)
    ax.set_xlabel(y_label)
    ax.legend(loc="upper left", shadow=True)
    plt.savefig(save_file_path)
    plt.close(fig)


if __name__ == "__main__":
    LEARNING_RATE = float(os.getenv("LEARNING_RATE"))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
    EPOCHS = int(os.getenv("EPOCHS"))
    DATA_PATH = os.getenv("DATA_PATH")
    MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH")

    epoch_list = []
    train_losses = []
    train_iou_scores = []
    train_dice_scores = []
    train_accuracy_scores = []

    val_losses = []
    val_iou_scores = []
    val_dice_scores = []
    val_accuracy_scores = []

    training_time = []

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = PlantSegmentationDataset(DATA_PATH)

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(
        train_dataset, [0.8, 0.2], generator=generator
    )

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_dataloader = DataLoader(
        dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    model = Unet(in_channels=3, num_classes=1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    criterion = nn.BCEWithLogitsLoss()
    accumulation_steps = 4

    start = time.time()

    for epoch in tqdm(range(EPOCHS)):
        epoch_list.append(epoch + 1)
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

            if (idx + 1) % accumulation_steps == 0 or (
                idx + 1 == len(train_dataloader)
            ):
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

        best_train_weights = are_best_weights(
            train_losses,
            train_iou_scores,
            train_dice_scores,
            train_accuracy_scores,
            train_loss,
            train_iou,
            train_dice,
            train_accuracy,
        )

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

        best_val_weights = are_best_weights(
            val_losses,
            val_iou_scores,
            val_dice_scores,
            val_accuracy_scores,
            val_loss,
            val_iou,
            val_dice,
            val_accuracy,
        )

        val_losses.append(val_loss)
        val_iou_scores.append(val_iou)
        val_dice_scores.append(val_dice)
        val_accuracy_scores.append(val_accuracy)

        training_time.append(time.time() - start)


        overall_weights = best_val_weights and best_train_weights

        print("-" * 30)
        print(
            f"EPOCH {epoch+1}/{EPOCHS}: Loss {train_loss:.4f}, "
            f"IoU {train_iou:.4f}, Dice {train_dice:.4f}, "
            f"Accuracy {train_accuracy:.4f}"
        )
        print("-" * 30)
        if (
            (epoch + 1) % 20 == 0
            or overall_weights
            or best_val_weights
            or best_train_weights
        ):
            if overall_weights:
                model_save_dir = f"{MODEL_SAVE_PATH}/best_weights_overall"
            elif best_val_weights:
                model_save_dir = f"{MODEL_SAVE_PATH}/best_weights_val"
            elif best_train_weights:
                model_save_dir = f"{MODEL_SAVE_PATH}/best_weights_train"
            else:
                model_save_dir = f"{MODEL_SAVE_PATH}/{epoch+1}_epochs"

            os.makedirs(model_save_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{model_save_dir}/unet.pt")

            actual = time.time()
            print(f"Training time: {actual-start:.2f}s")

            with open(f"{model_save_dir}/training_time.txt", "w") as f:
                f.write("Epoch: " + str(epoch+1) + "\n")
                f.write(f"Training time: {actual-start:.2f}s")

            generate_graph(
                train_losses,
                val_losses,
                epoch,
                "Loss/Error",
                "Epoch",
                "Loss",
                f"{model_save_dir}/loss.png",
            )
            generate_graph(
                train_iou_scores,
                val_iou_scores,
                epoch,
                "IoU",
                "Epoch",
                "IoU",
                f"{model_save_dir}/iou.png",
            )
            generate_graph(
                train_dice_scores,
                val_dice_scores,
                epoch,
                "Dice",
                "Epoch",
                "Dice",
                f"{model_save_dir}/dice.png",
            )
            generate_graph(
                train_accuracy_scores,
                val_accuracy_scores,
                epoch,
                "Accuracy",
                "Epoch",
                "Accuracy",
                f"{model_save_dir}/accuracy.png",
            )

    end = time.time()
    print(f"Total training time: {end-start:.2f}s")

    metrics_dict = {
        "epoch": epoch_list,
        "train_loss": train_losses,
        "train_iou": train_iou_scores,
        "train_dice": train_dice_scores,
        "train_accuracy": train_accuracy_scores,
        "val_loss": val_losses,
        "val_iou": val_iou_scores,
        "val_dice": val_dice_scores,
        "val_accuracy": val_accuracy_scores,
        "training_time": training_time,
    }
    pd.DataFrame(metrics_dict).to_csv(
        f'{MODEL_SAVE_PATH}/{datetime.now().strftime("%Y-%m-%d")}_training_{EPOCHS}_epochs_metrics.csv'
    )
