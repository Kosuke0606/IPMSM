"""
VGG16風の1チャネル入力カスタムモデルの訓練スクリプト
"""

import glob
import os
import random

import cupy as cp
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img


def main():
    # 再現性のための乱数シード設定
    SEED = 42

    # Python標準のrandomモジュールのシード設定
    random.seed(SEED)

    # NumPyのシード設定
    np.random.seed(SEED)

    # TensorFlowのシード設定
    tf.random.set_seed(SEED)

    # TensorFlowの決定的動作を有効化（可能な場合）
    # 注: 一部の演算では完全な決定性が保証されない場合があります
    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        # TensorFlow 2.9未満では利用できない
        pass

    # 環境変数でPythonハッシュのシードを固定
    os.environ["PYTHONHASHSEED"] = str(SEED)

    # CuPyのシード設定（GPU使用時）
    try:
        cp.random.seed(SEED)
    except:
        pass

    # 設定値
    IMG_BASE_PATH = "./20250425_data/Material_d_Gray/"
    CSV_BASE_PATH = "./20250425_data/TorqData/"
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    CHANNELS = 1  # グレースケール
    BATCH_SIZE = 128
    EPOCHS = 100
    LEARNING_RATE = 1e-4
    TEST_SPLIT_RATIO = 0.1
    VALID_SPLIT_RATIO = 0.1

    # CUDAデバイスの設定
    print("利用可能なGPUデバイス (Cupy):")
    try:
        for i in range(cp.cuda.runtime.getDeviceCount()):
            print(f"GPU {i}: {cp.cuda.runtime.getDeviceProperties(i)['name']}")
        if cp.cuda.runtime.getDeviceCount() > 0:
            cp.cuda.Device(0).use()
            print(
                f"\n選択されたCupy GPU: {cp.cuda.runtime.getDeviceProperties(0)['name']}"
            )
        else:
            print("\nCupyが利用できるGPUが見つかりませんでした。")
    except cp.cuda.runtime.CUDARuntimeError as e:
        print(f"Cupy CUDAエラー: {e}")

    print("\n利用可能なGPUデバイス (TensorFlow):")
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"{len(gpus)} 個のGPUがTensorFlowで利用可能です。")
        except RuntimeError as e:
            print(e)
    else:
        print("TensorFlowで利用可能なGPUが見つかりませんでした。CPUを使用します。")

    # CSVファイルから目的変数を抽出するヘルパー関数
    def get_target_value(csv_filepath):
        try:
            df = pd.read_csv(csv_filepath)
            target_df = df.head(30)
            if len(target_df) < 30:
                if len(target_df) == 0:
                    return np.nan
            target_mean = target_df["T"].mean()
            return target_mean
        except Exception:
            return np.nan

    # データ読み込みと準備
    image_files = sorted(glob.glob(os.path.join(IMG_BASE_PATH, "SingleI_*.bmp")))
    all_image_paths = []
    all_targets = []

    for img_path in image_files:
        basename = os.path.basename(img_path)
        try:
            parts_str = basename.replace("SingleI_", "").replace(".bmp", "")
            parts = parts_str.split("_")
            if len(parts) == 2:
                csv_filename = f"Data_{parts[0]}_{parts[1]}.csv"
                csv_filepath = os.path.join(CSV_BASE_PATH, csv_filename)
                if os.path.exists(csv_filepath):
                    target = get_target_value(csv_filepath)
                    if not np.isnan(target):
                        all_image_paths.append(img_path)
                        all_targets.append(target)
                    else:
                        print(
                            f"スキップ: {img_path} に対応する {csv_filepath} の目的変数が無効です。"
                        )
                else:
                    print(
                        f"警告: {img_path} に対応するCSVファイルが見つかりません: {csv_filepath}"
                    )
            else:
                print(f"警告: 画像ファイル名の形式が正しくありません: {img_path}")
        except Exception as e:
            print(f"エラー: 画像ファイル名 {img_path} の解析中にエラー: {e}")

    all_image_paths = np.array(all_image_paths)
    all_targets = np.array(all_targets)

    print(f"\n{len(all_image_paths)} 個の画像と対応する目的変数を読み込みました。")
    if len(all_image_paths) == 0:
        raise ValueError(
            "画像と目的変数のペアが読み込めませんでした。パスやファイル名、CSVの内容を確認してください。"
        )
    print(f"目的変数の例: {all_targets[:5]}")

    # データ分割
    train_val_paths, test_paths, train_val_targets, test_targets = train_test_split(
        all_image_paths,
        all_targets,
        test_size=TEST_SPLIT_RATIO,
        random_state=42,
        shuffle=True,
    )
    if len(train_val_paths) == 0:
        raise ValueError("テストデータ分割後、訓練・検証データが0になりました。")

    validation_split_from_train_val = VALID_SPLIT_RATIO / (1.0 - TEST_SPLIT_RATIO)
    if not (0 <= validation_split_from_train_val < 1):
        raise ValueError(
            f"検証データの分割比率が無効です: {validation_split_from_train_val}."
        )

    train_paths, val_paths, train_targets, val_targets = train_test_split(
        train_val_paths,
        train_val_targets,
        test_size=validation_split_from_train_val,
        random_state=42,
        shuffle=True,
    )

    print(f"訓練データ数: {len(train_paths)}")
    print(f"検証データ数: {len(val_paths)}")
    print(f"テストデータ数: {len(test_paths)}")

    if len(train_paths) == 0 or len(val_paths) == 0 or len(test_paths) == 0:
        print("警告: 訓練、検証、またはテストデータのいずれかが0件です。")

    # 1チャネル用の画像読み込みと前処理関数
    def load_and_preprocess_image_1channel(image_path_tensor):
        image_path_str = image_path_tensor.numpy().decode("utf-8")

        img = load_img(
            image_path_str,
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            color_mode="grayscale",
        )
        img_array = img_to_array(img)  # (height, width, 1)

        # 形状チェック
        if img_array.shape[-1] != 1:
            print(
                f"警告: {image_path_str} は1チャネルのはずが {img_array.shape} で読み込まれました。最初のチャネルを使用します。"
            )
            img_array = img_array[..., 0:1]

        # ピクセル値を0-1の範囲にスケーリング
        img_array_scaled = img_array / 255.0
        return tf.cast(img_array_scaled, tf.float32)

    # tf.data.Datasetの作成
    def create_dataset_1channel(paths, targets, batch_size, shuffle=True):
        if len(paths) == 0:
            print("警告: create_dataset に渡されたpathsの長さが0です。")
            return tf.data.Dataset.from_tensor_slices(
                (tf.constant([], dtype=tf.string), tf.constant([], dtype=tf.float32))
            ).batch(batch_size)

        path_ds = tf.data.Dataset.from_tensor_slices(paths)

        image_ds = path_ds.map(
            lambda x: tf.py_function(
                load_and_preprocess_image_1channel, [x], tf.float32
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

        def set_shape(img):
            img.set_shape((IMG_HEIGHT, IMG_WIDTH, CHANNELS))
            return img

        image_ds = image_ds.map(set_shape, num_parallel_calls=tf.data.AUTOTUNE)

        label_ds = tf.data.Dataset.from_tensor_slices(tf.cast(targets, tf.float32))
        ds = tf.data.Dataset.zip((image_ds, label_ds))

        if shuffle:
            # 再現性のためにseedを指定
            ds = ds.shuffle(
                buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=False
            )

        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
        return ds

    if len(train_paths) > 0:
        train_dataset = create_dataset_1channel(train_paths, train_targets, BATCH_SIZE)
    else:
        print("訓練データがありません。")
        train_dataset = None

    if len(val_paths) > 0:
        val_dataset = create_dataset_1channel(
            val_paths, val_targets, BATCH_SIZE, shuffle=False
        )
    else:
        print("検証データがありません。")
        val_dataset = None

    if len(test_paths) > 0:
        test_dataset = create_dataset_1channel(
            test_paths, test_targets, BATCH_SIZE, shuffle=False
        )
    else:
        print("テストデータがありません。")
        test_dataset = None

    print(
        f"Train dataset spec: {train_dataset.element_spec if train_dataset else 'None'}"
    )

    # モデルの定義
    def build_vgg_like_1channel_regressor(
        input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS),
        learning_rate_param=LEARNING_RATE,
    ):
        input_layer = Input(shape=input_shape, name="image_input")

        # Block 1
        x = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv1")(
            input_layer
        )
        x = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv2")(
            x
        )
        x = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv1")(
            x
        )
        x = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv2")(
            x
        )
        x = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv1")(
            x
        )
        x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv2")(
            x
        )
        x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv3")(
            x
        )
        x = MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv1")(
            x
        )
        x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv2")(
            x
        )
        x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv3")(
            x
        )
        x = MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv1")(
            x
        )
        x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv2")(
            x
        )
        x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv3")(
            x
        )
        x = MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)

        # 回帰のためのカスタム層
        x = GlobalAveragePooling2D(name="global_avg_pool")(x)
        x = Dense(512, activation="relu", name="fc1")(x)
        x = Dropout(0.5, name="dropout_fc1")(x)
        output_tensor = Dense(1, name="torque_output")(x)

        model = Model(
            inputs=input_layer,
            outputs=output_tensor,
            name="vgg_like_1channel_regressor",
        )

        optimizer = Adam(learning_rate=learning_rate_param)
        model.compile(
            optimizer=optimizer,
            loss="mean_squared_error",
            metrics=["mean_absolute_error", tf.keras.metrics.RootMeanSquaredError()],
        )
        return model

    model = build_vgg_like_1channel_regressor()
    model.summary()

    # 訓練時のコールバック設定
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
    )
    model_checkpoint = ModelCheckpoint(
        "best_vgg_like_1channel_regressor.keras",
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=5, min_lr=1e-6, verbose=1
    )
    callbacks_list = [early_stopping, model_checkpoint, reduce_lr]

    # モデルの訓練
    print("--- 初期訓練開始 (VGG16ベースは凍結) ---")
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks_list,
    )

    print("\n--- 訓練完了 ---")
    print(
        "最良のモデルは 'best_vgg_like_1channel_regressor.keras' として保存されました。"
    )

    return history


if __name__ == "__main__":
    main()
