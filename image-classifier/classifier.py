import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

# 学習率
lr = 0.001
batch_size = 32
# エポック数
num_epochs = 10

train_dir = "toyochem/train"
val_dir = "toyochem/val"

model_rev = "vgg16"

# 学習用の前処理
train_transform = transforms.Compose([
    transforms.Resize((100,100)),            # 256x256にリサイズ
    #transforms.CenterCrop((224,224)),        # 224x224にクロップ
    #transforms.RandomHorizontalFlip(p=0.5),  # 50%の確率で左右反転
    transforms.ToTensor(),                   # Tensorに変換（[0, 1]の範囲に正規化）
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # 平均と標準偏差で正規化
])

# 検証用とテスト用の前処理
val_transform = transforms.Compose([
    transforms.Resize((100,100)),
    #transforms.CenterCrop((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Datasetの作成
train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

# DataLoaderの作成
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

def train_epoch(model, dataloader, criterion, optimizer, device):
    # lossとaccの初期化
    train_loss, train_acc = 0, 0 # このtrain_lossとtrain_accにlossとaccを加算していき，最後にデータ数で割ることで平均を計算する

    model.train() # モデルを学習モードに設定
    for images, labels in tqdm(dataloader): # dataloaderからデータを取り出す
        images, labels = images.to(device), labels.to(device) # データをGPUに転送
        optimizer.zero_grad() # 勾配の初期化
        outputs = model(images) # モデルで推論
        loss = criterion(outputs, labels) # lossの計算
        train_loss += loss.item() * images.size(0) # lossを蓄積
        acc = (outputs.max(1)[1] == labels).sum() # accの計算
        train_acc += acc.item() # accを蓄積
        loss.backward() # 逆伝播
        optimizer.step() # パラメータの更新
    avg_train_loss = train_loss / len(dataloader.dataset) # lossの平均を計算
    avg_train_acc = train_acc / len(dataloader.dataset) # accの平均を計算
    return avg_train_loss, avg_train_acc

def val_epoch(model, dataloader, criterion, device):
    # lossとaccの初期化
    val_loss, val_acc = 0,0
    model.eval() # モデルを評価モードに設定
    with torch.no_grad(): # val_epoch関数では，勾配を計算しない
        for images, labels in tqdm(dataloader): # dataloaderからデータを取り出す
            images, labels = images.to(device), labels.to(device) # データをGPUに転送
            outputs = model(images) # モデルで推論
            loss = criterion(outputs, labels) # lossの計算
            val_loss += loss.item() * images.size(0) # lossを蓄積
            acc = (outputs.max(1)[1] == labels).sum() # accの計算
            val_acc += acc.item() # accを蓄積
    avg_val_loss = val_loss / len(dataloader.dataset) # lossの平均を計算
    avg_val_acc = val_acc / len(dataloader.dataset) # accの平均を計算
    return avg_val_loss, avg_val_acc

def set_trainable_layers(model, train_head_only=True):
    if train_head_only:
        # 全層凍結
        for param in model.parameters():
            param.requires_grad = False
        # ヘッドのみ学習可能にする
        for param in model.get_classifier().parameters():
            param.requires_grad = True
    else:
        # 全層学習可能にする
        for param in model.parameters():
            param.requires_grad = True


# クラス数の取得，確認
num_classes = len(train_dataset.classes)
print("クラス数: ", num_classes)

# 使用するGPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# モデルの作成
model = timm.create_model(model_rev, pretrained=True, num_classes=num_classes)
model.to(device)

# 損失関数
criterion = nn.CrossEntropyLoss()

# ヘッドのみ学習したい場合
set_trainable_layers(model, train_head_only=True)

# 全層微調整したい場合
# set_trainable_layers(model, train_head_only=False)

# optimizerは学習可能なパラメータだけ渡す
optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

# train_loss, train_acc, val_loss, val_accを保存するリスト
train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []

# 学習
for epoch in range(num_epochs):
    # train_epoch関数とval_epoch関数で1エポック分の学習と評価を行う
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
    
    # 各値をリストに追加
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)
    
    # ログを出力
    print(f"Epoch: {epoch+1}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}")

# modelの保存
#torch.save(model.state_dict(), f"model_epoch{num_epochs}.pth")
final_val_acc = val_acc_list[-1]
plt.figure(figsize=(15, 5))
plt.subplot(1,2,1)
plt.plot(train_loss_list, label="train_loss") # train_loss_listをプロットし，ラベルを指定
plt.plot(val_loss_list, label="val_loss") # val_loss_listをプロットし，ラベルを指定
plt.legend() # 凡例を表示
plt.subplot(1,2,2)
plt.plot(train_acc_list, label="train_acc")
plt.plot(val_acc_list, label="val_acc")
plt.legend()
plt.savefig(f"loss_{final_val_acc}_lr_{lr}batchsize_{batch_size}.png")
plt.show()


# 予測値と正解ラベルを格納するリスト
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in tqdm(val_loader):
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()  # 予測ラベル
        labels = labels.cpu().numpy()               # 正解ラベル

        all_preds.extend(preds)
        all_labels.extend(labels)

# 正解率計算
acc = accuracy_score(all_labels, all_preds)

# 混同行列の作成
cm = confusion_matrix(all_labels, all_preds)
class_names = val_dataset.classes  # クラス名を取得

# 混同行列の表示（seaborn）
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(f"matrix_{model_rev}-{num_epochs}-{acc:.2f}.png")
plt.show()
