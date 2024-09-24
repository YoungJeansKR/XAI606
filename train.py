import numpy as np
import pandas as pd 
import os
import random
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical, load_img
from sklearn.model_selection import train_test_split
import tensorflow as tf

print(tf.__version__)

# GPU 설정 확인 및 메모리 증가 허용
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # 모든 GPU에 메모리 증가 허용 설정
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("사용 가능한 GPU 수:", len(gpus))
    except RuntimeError as e:
        print(e)
else:
    print("GPU를 사용할 수 없습니다. CPU를 사용합니다.")

# 경로 설정
train_dir = "./input/train/train"

# 이미지 크기 및 채널 설정
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

# 파일 이름 및 카테고리 추출
filenames = os.listdir(train_dir)
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

# 데이터프레임 생성
df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

# 카테고리 문자열로 변환
df["category"] = df["category"].replace({0: 'cat', 1: 'dog'})

# 데이터 분할
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)

# 총 이미지 수 및 배치 크기 설정
total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size_per_gpu = 64  # 한 GPU당 배치 크기 설정

# 멀티 GPU를 위한 전략 설정
# NCCL을 지원하지 않는 Windows 환경에서는 cross_device_ops를 설정해야 합니다.
strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
)
print('사용 중인 디바이스 수:', strategy.num_replicas_in_sync)

global_batch_size = batch_size_per_gpu * strategy.num_replicas_in_sync

with strategy.scope():
    # 이미지 데이터 제너레이터 설정
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        rescale=1./255,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    train_dir, 
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='categorical',
    batch_size=global_batch_size,
    workers=8,  # 병렬 처리할 워커 수
    use_multiprocessing=True  # 다중 프로세싱 사용
    )

    validation_datagen = ImageDataGenerator(rescale=1./255)
    validation_generator = validation_datagen.flow_from_dataframe(
        validate_df, 
        train_dir, 
        x_col='filename',
        y_col='category',
        target_size=IMAGE_SIZE,
        class_mode='categorical',
        batch_size=global_batch_size  # 배치 크기 수정
    )

    # 모델 구성
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(0.25))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

    # 모델 컴파일
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# 콜백 설정
earlystop = tf.keras.callbacks.EarlyStopping(patience=10)
learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', 
                                                               patience=2, 
                                                               verbose=1, 
                                                               factor=0.5, 
                                                               min_lr=0.00001)
callbacks = [earlystop, learning_rate_reduction]

# 스텝 수 계산
steps_per_epoch = total_train // global_batch_size
validation_steps = total_validate // global_batch_size

# 모델 훈련
epochs = 1
history = model.fit(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    steps_per_epoch=steps_per_epoch,
    callbacks=callbacks
)

# 모델 저장
model.save_weights("model.h5")
