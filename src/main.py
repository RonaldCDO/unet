import torch
from torch import optim, nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from unet import Unet
from dataset import PlantSegmentationDataset
import os
from dotenv import load_dotenv

torch.cuda.empty_cache()
 
load_dotenv()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

scaler = torch.cuda.amp.GradScaler()

if __name__ == "__main__":
    LEARNING_RATE=os.getenv('LEARNING_RATE')
    BATCH_SIZE=os.getenv('BATCH_SIZE')
    EPOCHS=os.getenv('EPOCHS')
    DATA_PATH=os.getenv('DATA_PATH')
    MODEL_SAVE_PATH=os.getenv('MODEL_SAVE_PATH')

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

    for epoch in tqdm(range(EPOCHS)):
        model.train()
        train_running_loss = 0
        for idx, img_mask in enumerate(tqdm(train_dataloader)):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].float().to(device)

            with torch.cuda.amp.autocast_mode.autocast():
                y_pred = model(img)
                loss = criterion(y_pred, mask)
                train_running_loss += loss.item()

            loss = loss/accumulation_steps
            scaler.scale(loss).backward()

            if (idx + 1) % accumulation_steps == 0 or (idx + 1 == len(train_dataloader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        train_loss = train_running_loss / (idx + 1)

        model.eval()
        val_running_loss = 0
        with torch.no_grad():
            for idx, img_mask in enumerate(tqdm(val_dataloader)):
                img = img_mask[0].float().to(device)
                mask = img_mask[1].float().to(device)

                y_pred = model(img)
                loss = criterion(y_pred, mask)
                val_running_loss += loss.item()

        val_loss = val_running_loss / (idx + 1)

        print("-"*30)
        print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
        print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
        print("-"*30)
        if epoch % 20 == 0:
            torch.save(model.state_dict(), MODEL_SAVE_PATH+f'unet-{epoch}.pt')
