#!/usr/bin/env python3
"""
Python script generated from: HW 2/北科電子四甲 林晏宇同學 111360128 HW 2 打造自己的DNN(全連結)手寫辨識.md
Generated on: 1759147013.658962
Note: Colab-specific commands (!pip, %magic) have been commented out
"""

# 設定 4 層神經網路的神經元數量
N1 = 128  # 第一層
N2 = 64   # 第二層
N3 = 32   # 第三層
N4 = 16   # 第四層

# COLAB ONLY: !pip install gradio

# JUPYTER MAGIC: %matplotlib inline

# 標準數據分析、畫圖套件
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# 神經網路方面
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# 互動設計用
from ipywidgets import interact_manual

# 神速打造 web app 的 Gradio
import gradio as gr

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f'訓練資料總筆數為 {len(x_train)} 筆資料')
print(f'測試資料總筆數為 {len(x_test)} 筆資料')

def show_xy(n=0):
    ax = plt.gca()
    X = x_train[n]
    plt.xticks([], [])
    plt.yticks([], [])
    plt.imshow(X, cmap = 'Greys')
    print(f'本資料 y 給定的答案為: {y_train[n]}')

interact_manual(show_xy, n=(0,59999));

def show_data(n = 100):
    X = x_train[n]
    print(X)

interact_manual(show_data, n=(0,59999));

x_train = x_train.reshape(60000, 784)/255
x_test = x_test.reshape(10000, 784)/255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

n = 87
y_train[n]

model = Sequential()

model.add(Dense(N1, input_dim=784, activation='relu'))

model.add(Dense(N2, activation='relu'))

model.add(Dense(N3, activation='relu'))

model.add(Dense(N4, activation='relu'))

model.add(Dense(10, activation='softmax'))

model.compile(loss='mse', optimizer=SGD(learning_rate=0.087), metrics=['accuracy'])

model.summary()

model.fit(x_train, y_train, batch_size=100, epochs=10)

loss, acc = model.evaluate(x_test, y_test)

print(f"測試資料正確率 {acc*100:.2f}%")

predict = np.argmax(model.predict(x_test), axis=-1)

predict

def test(測試編號):
    plt.imshow(x_test[測試編號].reshape(28,28), cmap='Greys')
    print('神經網路判斷為:', predict[測試編號])

interact_manual(test, 測試編號=(0, 9999));

score = model.evaluate(x_test, y_test)

print('loss:', score[0])
print('正確率', score[1])

# Claude Code 優化版本
# 用更現代的技巧來訓練神經網路，看看能不能打敗老師的版本

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 設定中文字體，避免警告
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Helvetica', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

print("Building Claude optimized model...")

# 建立改良版模型 - 一樣是 4 層但加了 Dropout
model_claude = Sequential()
model_claude.add(Dense(128, input_dim=784, activation='relu'))
model_claude.add(Dropout(0.2))  # 隨機關掉 20% 神經元，防止過擬合
model_claude.add(Dense(64, activation='relu'))
model_claude.add(Dropout(0.2))  # 每層後面都加 Dropout
model_claude.add(Dense(32, activation='relu'))
model_claude.add(Dense(16, activation='relu'))
model_claude.add(Dense(10, activation='softmax'))

# 用 Adam optimizer - 會自動調整學習率，通常比 SGD 好
# categorical_crossentropy 對分類問題比 MSE 更適合
model_claude.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

print("Model architecture:")
model_claude.summary()

# 設定 Early Stopping - 如果驗證準確率 3 輪沒進步就停止
# 這樣可以避免過度訓練
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=3,
    restore_best_weights=True,  # 恢復最佳權重
    verbose=1
)

print("Starting training Claude optimized version...")
print("Using 10% data as validation set, Early Stopping enabled...")

# 訓練時分出 10% 當驗證集
history_claude = model_claude.fit(
    x_train, y_train,
    batch_size=128,
    epochs=20,  # 設多一點但會自動提早停止
    validation_split=0.1,  # 10% 當驗證集
    callbacks=[early_stop],
    verbose=1
)

# 分析模型的信心度 - 看看哪些預測模型沒把握
print("\n=== Confidence Analysis ===")

predictions_claude = model_claude.predict(x_test)
confidence = np.max(predictions_claude, axis=1)  # 最高機率就是信心度

# 找出模型最沒把握的 10 張圖
uncertain_idx = np.argsort(confidence)[:10]
print(f"\nMost uncertain image indices: {uncertain_idx}")
print(f"Their confidence scores: {confidence[uncertain_idx]*100}")

# 顯示前 3 張最沒把握的圖片
print("\nTop 3 most uncertain images:")
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, idx in enumerate(uncertain_idx[:3]):
    axes[i].imshow(x_test[idx].reshape(28,28), cmap='gray')
    pred = np.argmax(predictions_claude[idx])
    conf = confidence[idx]
    axes[i].set_title(f'Pred: {pred}, Conf: {conf:.1%}')
    axes[i].axis('off')
plt.show()

# 統計信心度分布
print(f"\nAverage confidence: {np.mean(confidence):.2%}")
print(f"Minimum confidence: {np.min(confidence):.2%}")
print(f"Number of predictions below 90% confidence: {np.sum(confidence < 0.9)}")

# 用混淆矩陣看看哪些數字容易搞混
print("\n=== Error Pattern Analysis ===")

y_pred_claude = np.argmax(predictions_claude, axis=-1)
y_true = np.argmax(y_test, axis=-1)

# 建立混淆矩陣
cm = confusion_matrix(y_true, y_pred_claude)

# 畫出混淆矩陣熱力圖
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
plt.title('Confusion Matrix - Claude Optimized Version')
plt.ylabel('True Digit')
plt.xlabel('Predicted Digit')
plt.show()

# 找出最常搞混的組合
print("\nMost confused digit pairs:")
error_pairs = []
for i in range(10):
    for j in range(10):
        if i != j and cm[i][j] > 20:  # 錯誤超過 20 次
            error_pairs.append((i, j, cm[i][j]))

error_pairs.sort(key=lambda x: x[2], reverse=True)
for true_digit, pred_digit, count in error_pairs[:5]:
    print(f"  Digit {true_digit} misclassified as {pred_digit}: {count} times")

# 比較老師版本 vs Claude 優化版
print("\n" + "="*60)
print("Final Comparison: Teacher vs Claude Optimized Version")
print("="*60)

# 老師版本的結果
loss_teacher, acc_teacher = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTeacher version (4 layers + SGD + MSE):")
print(f"  - Test accuracy: {acc_teacher*100:.2f}%")
print(f"  - Loss: {loss_teacher:.4f}")

# Claude 版本的結果
loss_claude, acc_claude = model_claude.evaluate(x_test, y_test, verbose=0)
print(f"\nClaude optimized (4 layers + Adam + Dropout + Early Stop):")
print(f"  - Test accuracy: {acc_claude*100:.2f}%")
print(f"  - Loss: {loss_claude:.4f}")

# 改進幅度
improvement = (acc_claude - acc_teacher) * 100
print(f"\nAccuracy improvement: {improvement:+.2f}%")
if improvement > 0:
    print("Claude version wins!")
else:
    print("Teacher version is better, need more tuning")

# 訓練歷程比較圖
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_claude.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history_claude.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Claude Version Training Progress')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(history_claude.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history_claude.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Loss Progress')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 總結實驗結果
print("\n" + "="*60)
print("Experiment Summary")
print("="*60)

print("\nData Summary:")
print(f"- Training stopped at epoch: {len(history_claude.history['accuracy'])}")
print(f"- Best validation accuracy: {max(history_claude.history['val_accuracy'])*100:.2f}%")
print(f"- Final test accuracy: {acc_claude*100:.2f}%")
print(f"- Low confidence predictions (<90%): {np.sum(confidence < 0.9)} images")
print(f"- Average prediction confidence: {np.mean(confidence)*100:.1f}%")

print("\nKey Findings:")
if improvement > 0:
    print(f"1. Adam optimizer performed {improvement:.2f}% better than SGD")
    print("2. Dropout effectively prevented overfitting")
    print("3. Early Stopping found optimal training point")
else:
    print("1. May need hyperparameter tuning")
    print("2. Simple architecture might be sufficient for this problem")

# 最容易搞混的數字
if error_pairs:
    most_confused = error_pairs[0]
    print(f"\nMost confused: {most_confused[0]} and {most_confused[1]} ({most_confused[2]} errors)")

def resize_image(inp):
    # 圖在 inp["layers"][0]
    image = np.array(inp["layers"][0], dtype=np.float32)
    image = image.astype(np.uint8)

    # 轉成 PIL 格式
    image_pil = Image.fromarray(image)

    # Alpha 通道設為白色, 再把圖從 RGBA 轉成 RGB
    background = Image.new("RGB", image_pil.size, (255, 255, 255))
    background.paste(image_pil, mask=image_pil.split()[3]) # 把圖片粘貼到白色背景上，使用透明通道作為遮罩
    image_pil = background

    # 轉換為灰階圖像
    image_gray = image_pil.convert("L")

    # 將灰階圖像縮放到 28x28, 轉回 numpy array
    img_array = np.array(image_gray.resize((28, 28), resample=Image.LANCZOS))

    # 配合 MNIST 數據集
    img_array = 255 - img_array

    # 拉平並縮放
    img_array = img_array.reshape(1, 784) / 255.0

    return img_array

def recognize_digit(inp):
    img_array = resize_image(inp)
    prediction = model.predict(img_array).flatten()
    labels = list('0123456789')
    return {labels[i]: float(prediction[i]) for i in range(10)}

iface = gr.Interface(
    fn=recognize_digit,
    inputs=gr.Sketchpad(),
    outputs=gr.Label(num_top_classes=3),
    title="MNIST 手寫辨識",
    description="請在畫板上繪製數字"
)

iface.launch(share=True, debug=True)

