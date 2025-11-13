import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from timm.data import resolve_model_data_config

matplotlib.use('Agg')  # 画面表示なしのバックエンドに切り替え


# GUI部分
class GUIApp:
    def __init__(self, master):
        self.master = master
        master.title("PyTorch 分類モデル 学習GUI")
        master.geometry("600x500")  # ウィンドウサイズ指定
        master.grid_columnconfigure(0, weight=1)
        master.grid_columnconfigure(1, weight=1)
        self.stop_flg = False

        self.train_dir = ""
        self.val_dir = ""
        
        self.model_entry = self.add_entry(0, "モデル名", "vgg16")
        self.lr_entry = self.add_entry(1, "学習率", "0.001")
        self.epoch_entry = self.add_entry(2, "エポック数", "1")
        self.batch_entry = self.add_entry(3, "バッチサイズ", "16")
        self.resize_entry = self.add_entry(4, "画像サイズ (例: 100x100)", "100x100")
        
        # 学習データフォルダ選択ボタンとその下のパス表示ラベル
        tk.Button(master, text="学習データフォルダ選択", command=self.select_train).grid(row=7, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 2))
        self.train_label = tk.Label(master, text="未選択", fg="gray")
        self.train_label.grid(row=8, column=0, columnspan=2, sticky="w", padx=20)
        tk.Button(master, text="検証データフォルダ選択", command=self.select_val).grid(row=9, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 2))
        self.val_label = tk.Label(master, text="未選択", fg="gray")
        self.val_label.grid(row=10, column=0, columnspan=2, sticky="w", padx=20)
        self.head_only = tk.BooleanVar(value=True)
        tk.Checkbutton(master, text="ヘッドのみ学習", variable=self.head_only).grid(row=11, column=0, columnspan=2)
        self.use_pretrained = tk.BooleanVar(value=True)
        tk.Checkbutton(master, text="学習済み重みを使用（pretrained=True）", variable=self.use_pretrained).grid(row=12, column=0, columnspan=2)
        # EarlyStopping チェックボックス用の変数
        self.early_stopping_var = tk.BooleanVar(value=False)
        tk.Checkbutton(master, text="EarlyStopping を有効にする", variable=self.early_stopping_var).grid(row=13, column=0, columnspan=2)
        self.status = tk.Label(master, text="準備完了")
        self.status.grid(row=14, column=0, columnspan=2)
        tk.Button(master,text="学習開始",command=self.start_training).grid(row=15, column=0, columnspan=2)
        tk.Button(master,text="中断",command=self.stop_training).grid(row=16, column=0, columnspan=2)

    def add_entry(self, row, label, default=""):
        tk.Label(self.master, text=label).grid(row=row, column=0, sticky="e", padx=5, pady=5)
        entry = tk.Entry(self.master)
        entry.insert(0, default)
        entry.grid(row=row, column=1, sticky="w", padx=5, pady=5)
        return entry
        
    def select_train(self):
        path = filedialog.askdirectory()
        if path:  # ← 空でなければ更新
            self.train_dir = path
            self.train_label.config(text=path, fg="black")

    def select_val(self):
        path = filedialog.askdirectory()
        if path:
            self.val_dir = path
            self.val_label.config(text=path, fg="black")

    def start_training(self):
        self.stop_flg = False
        threading.Thread(target=self.train).start()

    def train(self) -> None:
        # 画像サイズパース
        resize_str = self.resize_entry.get()  # "100x100"
        try:
            resize_tuple = tuple(map(int, resize_str.lower().split("x")))
            if len(resize_tuple) != 2:
                raise ValueError("画像サイズは '幅x高さ' の形式で指定してください。")
        except Exception:
            messagebox.showerror("エラー", "画像サイズの形式が不正です（例: 100x100）")
            return
        try:
            self.master.after(0, lambda: self.status.config(text="学習中..."))
            self.run_training(
                model_rev=self.model_entry.get(),
                lr=float(self.lr_entry.get()),
                num_epochs=int(self.epoch_entry.get()),
                batch_size=int(self.batch_entry.get()),
                train_dir=self.train_dir,
                val_dir=self.val_dir,
                head_only=self.head_only.get(),
                resize_tuple=resize_tuple,
                pretrained=self.use_pretrained.get(),
                should_stop_func=lambda: self.stop_flg,
                use_early_stopping=self.early_stopping_var.get(),
                patience=5
            )
            self.master.after(0, lambda: self.status.config(text="完了"))
        except Exception as e:
            self.show_error(e)

    def show_error(self, err: Exception):
        messagebox.showerror("エラー", str(err))

    def stop_training(self) -> None:
        self.stop_flg = True
        self.status.config(text="中止指示済み（変更可）")

    def on_closing(self) -> None:
        plt.close('all')  # ウィンドウ閉じる前に全グラフ閉じる
        self.master.destroy()

    def run_training(self,
                     model_rev: str,
                     lr: float,
                     num_epochs: int,
                     batch_size: int,
                     train_dir: str,
                     val_dir: str,
                     head_only: bool,
                     resize_tuple: tuple,
                     pretrained: bool = True,
                     should_stop_func = None,
                     use_early_stopping: bool = True,
                     patience: int = 10
                     ) -> None:
        train_loss_list, train_acc_list = [], []
        val_loss_list, val_acc_list = [], []
        best_acc = 0.0
        best_epoch = 0
        epochs_no_improve = 0
        pth_dir = "results/pth"
        pth_path = ""

        if not train_dir or not os.path.exists(train_dir):
            messagebox.showerror("エラー", "学習フォルダが設定されていません")
            return
        if len(os.listdir(train_dir)) < 2:
            messagebox.showerror("エラー", "学習フォルダにクラスごとのサブフォルダが2つ以上必要です")
            return
        model = timm.create_model(model_rev, pretrained=pretrained)
        config = resolve_model_data_config(model)
        mean, std = config["mean"], config["std"]
        data_transforms = transforms.Compose([
            transforms.Resize(resize_tuple),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
            ])
        train_dataset = datasets.ImageFolder(train_dir, transform=data_transforms)
        val_dataset = datasets.ImageFolder(val_dir, transform=data_transforms)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        num_classes = len(train_dataset.classes)
        model = timm.create_model(model_rev, pretrained=pretrained, num_classes=num_classes) 
        model.to(device)
        self.set_trainable_layers(model, head_only)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        os.makedirs(pth_dir, exist_ok=True)
        for epoch in range(num_epochs):
            if should_stop_func and should_stop_func():
                print("⚠️ 学習が中断されました")
                return
            train_loss, train_acc = self.train_epoch(model, train_loader, criterion, optimizer, device, should_stop_func=lambda: self.stop_flg,)
            val_loss, val_acc = self.val_epoch(model, val_loader, criterion, device, should_stop_func=lambda: self.stop_flg,)
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            val_loss_list.append(val_loss)
            val_acc_list.append(val_acc)
            # ✅ ベストモデル保存ロジック
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                pth_path = os.path.join(pth_dir, f"model_{model_rev}.pth")
                torch.save(model.state_dict(), pth_path)
                print(
                    f"✅ Saved best model at epoch {epoch} "
                    f"with acc={val_acc:.2f}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            print(
                f"[{epoch+1}/{num_epochs}] "
                f"Train Loss: {train_loss:.4f}, "
                f"Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, "
                f"Acc: {val_acc:.4f}"
                )
            # ✅ EarlyStopping 発動条件
            if use_early_stopping and epochs_no_improve >= patience:
                print(
                    f"⏹ EarlyStopping 発動 (epoch {epoch+1}) - "
                    f"best acc={best_acc:.2f} (epoch {best_epoch+1})"
                    )
                break
        head_flag = "head_only=True" if head_only else "head_only=False"
        if len(val_acc_list) == 0:
            raise ValueError("学習が1エポックも行われていません。設定を確認してください。")
        final_val_acc = val_acc_list[-1]
        loss_dir = "results/loss"
        os.makedirs(loss_dir, exist_ok=True)
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss_list, label="train_loss")
        plt.plot(val_loss_list, label="val_loss")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(train_acc_list, label="train_acc")
        plt.plot(val_acc_list, label="val_acc")
        plt.legend()
        plt.title(
            f"{model_rev}-Loss | lr={lr}, batch={batch_size},"
            f"epochs={num_epochs}\n{head_flag}"
            )
        # --- loss曲線の保存 ---
        loss_path = self.get_unique_filepath(
            os.path.join(loss_dir, f"loss_{model_rev}_{final_val_acc*100:.2f}.png")
            )
        plt.savefig(loss_path)
        print(f"保存しました。{loss_path}")
        # 混同行列
        all_preds, all_labels = [], []
        model.load_state_dict(torch.load(pth_path))
        model.eval()
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Predict"):
                images = images.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1).cpu().numpy()
                labels = labels.cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels)

        acc = accuracy_score(all_labels, all_preds)
        cm = confusion_matrix(all_labels, all_preds)
        class_names = val_dataset.classes
        matrix_dir = "results/matrix"
        os.makedirs(matrix_dir, exist_ok=True)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(
            f"{model_rev}-Matrix | acc={acc*100:.2f}, lr={lr}, "
            f"batch={batch_size}, epochs={num_epochs}\n{head_flag}"
            )
        plt.title(
            f"{model_rev}-Matrix | acc={acc*100:.2f}, lr={lr}, "
            f"batch={batch_size}, epochs={num_epochs}\n{head_flag}"
            )

        plt.tight_layout()
        matrix_path = self.get_unique_filepath(
            os.path.join(matrix_dir, f"matrix_{model_rev}_acc{acc*100:.2f}.png")
            )
        plt.savefig(matrix_path)
        print(f"保存しました{matrix_path}")

        excel_file = "output.xlsx"
        result = {
            "モデル名": model_rev,
            "学習率": lr,
            "エポック数": num_epochs,
            "バッチサイズ": batch_size,
            "精度(acc)": round(final_val_acc * 100, 2),  # % にして保存
            "画像サイズ": f"{resize_tuple[0]}x{resize_tuple[1]}",
            "Normalize平均": ",".join(map(str, mean)),
            "Normalize分散": ",".join(map(str, std)),
            "学習": head_flag,
            "pretrained": pretrained
        }

        if os.path.exists(excel_file):
            # 既存ファイルに追記
            df = pd.read_excel(excel_file)
            new_row = pd.DataFrame([result])
            df = pd.concat([df, new_row], ignore_index=True)

        else:
            # 新規作成
            df = pd.DataFrame([result])

        df.to_excel(excel_file, index=False)
        print(f"結果を {excel_file} に保存しました。")
    
    def set_trainable_layers(self, model, train_head_only=True):
        if train_head_only:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.get_classifier().parameters():
                param.requires_grad = True
        else:
            for param in model.parameters():
                param.requires_grad = True

    def train_epoch(self, model, dataloader, criterion, optimizer, device, should_stop_func = False):
        model.train()
        train_loss, train_acc = 0, 0
        for images, labels in tqdm(dataloader, desc="Training"):
            if should_stop_func and should_stop_func():
                print("⚠️ 学習が中断されました")
                return
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            train_loss += loss.item() * images.size(0)
            acc = (outputs.max(1)[1] == labels).sum()
            train_acc += acc.item()
            loss.backward()
            optimizer.step()
        return (
            train_loss / len(dataloader.dataset),
            train_acc / len(dataloader.dataset),
        )

    def val_epoch(self, model, dataloader, criterion, device, should_stop_func = False):
        model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Validation"):
                if should_stop_func and should_stop_func():
                    print("⚠️ 学習が中断されました")
                    return
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                acc = (outputs.max(1)[1] == labels).sum()
                val_acc += acc.item()
        return (
            val_loss / len(dataloader.dataset),
            val_acc / len(dataloader.dataset),
        )

    def get_unique_filepath(self, base_path):
        """
        base_path にファイルパスを入れて、もし同名ファイルがあれば
        _1, _2, ... のように連番を付けて重複しないファイルパスを返す
        """
        if not os.path.exists(base_path):
            return base_path

        base, ext = os.path.splitext(base_path)
        i = 1
        while True:
            new_path = f"{base}_{i}{ext}"
            if not os.path.exists(new_path):
                return new_path
            i += 1


if __name__ == "__main__":
    root = tk.Tk()
    app = GUIApp(root)
    root.mainloop()
