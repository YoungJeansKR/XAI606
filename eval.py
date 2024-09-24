import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import load_img
import tensorflow as tf

# 경로 설정
test_dir = "./input/test1/test1"
train_dir = "./input/train/train"

# 이미지 크기 및 채널 설정
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3

# 배치 크기 설정 (한 GPU당 배치 크기)
batch_size_per_gpu = 32  # 훈련 때 사용했던 값과 동일하게 설정

# 멀티 GPU를 위한 전략 설정
strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
)
print('사용 중인 디바이스 수:', strategy.num_replicas_in_sync)

# 전체 배치 크기 설정
global_batch_size = batch_size_per_gpu * strategy.num_replicas_in_sync

# 테스트 데이터 준비
test_filenames = os.listdir(test_dir)
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]

# 이미지 데이터 제너레이터 설정 (테스트 데이터의 경우 rescale만 적용)
test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    test_dir, 
    x_col='filename',
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=global_batch_size,  # 멀티 GPU에 맞춘 배치 크기
    shuffle=False
)

# 멀티 GPU를 위한 전략 설정
strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
)

with strategy.scope():
    # 훈련 시 사용한 모델과 동일한 구조로 모델 생성
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

    # 모델 컴파일 (훈련 시 사용한 것과 동일하게 설정)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# 이전에 저장한 가중치 로드
model.load_weights("model.h5")

# 예측 수행
steps = np.ceil(nb_samples / global_batch_size)
predict = model.predict(test_generator, steps=steps)

# 예측 결과 처리
test_df['category'] = np.argmax(predict, axis=-1)

# 클래스 인덱스 맵핑
label_map = {0: 'cat', 1: 'dog'}
test_df['category'] = test_df['category'].replace(label_map)

# 결과 시각화
test_df['category'].value_counts().plot.bar()
plt.show()

# 샘플 이미지 출력
sample_test = test_df.head(18)
plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['filename']
    category = row['category']
    img = load_img(os.path.join(test_dir, filename), target_size=IMAGE_SIZE)
    plt.subplot(6, 3, index+1)
    plt.imshow(img)
    plt.xlabel(f"{filename} ({category})")
plt.tight_layout()
plt.show()

# 제출 파일 생성
submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = submission_df['category'].replace({'dog': 1, 'cat': 0})
submission_df.drop(['filename', 'category'], axis=1, inplace=True)
submission_df.to_csv('submission.csv', index=False)
