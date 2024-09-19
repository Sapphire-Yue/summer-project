import os
from sgfmill import sgf, boards

# 讀取多個SGF
sgf_folder_path = "sgf/"
sgf_files = [f for f in os.listdir(sgf_folder_path) if (f.endswith('.SGF') or f.endswith('.sgf'))]
sgf_games = []

for sgf_file in sgf_files:
    sgf_file_path = os.path.join(sgf_folder_path, sgf_file)
    
    # 讀取棋譜
    with open(sgf_file_path, "rb") as f:
        sgf_content = f.read()

    game = sgf.Sgf_game.from_bytes(sgf_content)
    sgf_games.append(game)
    print(f"導入成功: {sgf_file}")
    

#---------------------------------------------------------------------

import numpy as np

def board_to_matrix(board_history, current_player_color):
    """
    將棋盤歷史轉換為 17 個特徵平面的矩陣。
    board_history: 包含過去 7 步棋盤狀態的列表，每個元素都是一個 Board 對象
    current_player_color: 當前玩家的顏色，'b' 或 'w'
    return shape 為 (19, 19, 17) 的 numpy 數組
    """
    board_size = board_history[0].side
    feature_planes = []

    # 添加過去 7 步的棋盤狀態
    for board in board_history:
        black_plane = np.zeros((board_size, board_size), dtype=np.int8)
        white_plane = np.zeros((board_size, board_size), dtype=np.int8)
        for row in range(board_size):
            for col in range(board_size):
                stone = board.get(row, col)
                if stone == 'b':
                    black_plane[row, col] = 1
                elif stone == 'w':
                    white_plane[row, col] = 1
        feature_planes.append(black_plane)
        feature_planes.append(white_plane)


    # 添加當前玩家特徵平面
    if current_player_color == 'b':
        player_plane = np.ones((board_size, board_size), dtype=np.int8)
    else:
        player_plane = np.zeros((board_size, board_size), dtype=np.int8)
    feature_planes.append(player_plane)

    # 堆疊成形狀為 (19, 19, 17) 的數組
    feature_planes = np.stack(feature_planes, axis=-1)
    return feature_planes


#---------------------------------------------------------------------

def move_to_label(row, col, board_size=19):
    return row * board_size + col

def normalize_board(matrix):
    return matrix.astype(np.float32)  # 將整數矩陣轉為浮點數

def rotate_board(matrix):
    return np.rot90(matrix)

def flip_board(matrix):
    return np.fliplr(matrix)  # 左右翻轉

#---------------------------------------------------------------------

x_train = []  # 保存棋盤狀態
y_train = []  # 保存對應的動作標籤

def rotate_action(action_label, k, board_size=19):
    row = action_label // board_size
    col = action_label % board_size
    for _ in range(k):
        row, col = col, board_size - 1 - row
    return row * board_size + col

def flip_action(action_label, board_size=19):
    row = action_label // board_size
    col = action_label % board_size
    col = board_size - 1 - col
    return row * board_size + col


for game in sgf_games:
    board_size = game.get_size()
    board = boards.Board(board_size)
    board_history = [board.copy()] * 8  # 初始化過去 7 步棋盤狀態（包含當前狀態）
    for move in game.main_sequence_iter():
        move_info = move.get_move()
        if move_info[0] is not None:
            color, (row, col) = move_info

            # 構建特徵平面
            current_player_color = color
            feature_planes = board_to_matrix(board_history, current_player_color)
            x_train.append(feature_planes)

            # 動作標籤
            action_label = move_to_label(row, col)
            y_train.append(action_label)

            # 數據增強（旋轉和翻轉）
            for k in range(1, 4):
                augmented_planes = np.rot90(feature_planes, k=k, axes=(0, 1))
                x_train.append(augmented_planes)
                y_train.append(rotate_action(action_label, k))

            flipped_planes = np.flip(feature_planes, axis=1)
            x_train.append(flipped_planes)
            y_train.append(flip_action(action_label))

            # 更新棋盤歷史
            board.play(row, col, color)
            board_history = board_history[1:] + [board.copy()]

x_train = np.array(x_train)  # 形狀應該是 (樣本數, 19, 19, 17)
y_train = np.array(y_train)  # 形狀應該是 (樣本數,)


#---------------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras import layers

y_train = tf.keras.utils.to_categorical(y_train, num_classes=19 * 19)
print(f"x_train.shape: {x_train.shape}")
print(f"y_train.shape: {y_train.shape}")

# 定義一個殘差塊 (ResNet block)
def residual_block(x, filters, kernel_size=3):
    # 卷積層 1
    residual = x
    x = layers.Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 卷積層 2
    x = layers.Conv2D(filters, kernel_size=kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # 殘差連接
    x = layers.Add()([x, residual])
    x = layers.ReLU()(x)
    return x

# 定義 ResNet 模型的主體
def create_resnet(input_shape, num_blocks=20, filters=256):
    inputs = tf.keras.Input(shape=input_shape)
    
    # 初始卷積層
    x = layers.Conv2D(filters, kernel_size=3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # 添加指定數量的殘差塊
    for _ in range(num_blocks):
        x = residual_block(x, filters)

    return inputs, x

# 策略網路（輸出下一步棋的概率分佈）
def policy_head(x, board_size=19):
    x = layers.Conv2D(2, kernel_size=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(board_size * board_size, activation='softmax')(x)  # 361 個位置的概率分佈
    return x

# 價值網路（輸出當前局面勝率）
def value_head(x):
    x = layers.Conv2D(1, kernel_size=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dense(1, activation='tanh')(x)  # 輸出一個勝率標量，範圍[-1, 1]
    return x

# 創建完整模型
def create_mcts_resnet_model(input_shape, num_blocks=20, filters=256, board_size=19):
    inputs, backbone = create_resnet(input_shape, num_blocks, filters)
    
    # 策略與價值頭
    policy = policy_head(backbone, board_size)
    value = value_head(backbone)
    
    model = tf.keras.Model(inputs=inputs, outputs=[policy, value])
    return model

# 初始化模型
input_shape = (19, 19, 17)  # 棋盤狀態的輸入
model = create_mcts_resnet_model(input_shape)
epochs = 50
model.summary()

# 編譯模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=['categorical_crossentropy', 'mean_squared_error'],
              metrics=['accuracy'])

# 將 x_train 和 y_train 填充進去進行訓練
history = model.fit(x_train, y_train, epochs=epochs, validation_split=0.3)
print(history)

#---------------------------------------------------------------------
import matplotlib.pyplot as plt

plt.title('Training Accuracy')

plt.plot(history.history['dense_accuracy'], label='accuracy')
plt.plot(history.history['val_dense_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.01, 1.0])
plt.legend(loc='lower right')

# 顯示圖表
plt.show()

model.save('ResNet_MCTS.h5')