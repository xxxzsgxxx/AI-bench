#!/bin/bash
# AI框架测试模块
set -e

LOG_DIR="../sysinfo_$(date +%Y%m%d_%H%M%S)/ai_tests"
mkdir -p ${LOG_DIR}/{tensorflow,pytorch}

prepare_env() {
    source .venv/bin/activate
    echo "[环境检查]" > ${LOG_DIR}/environment.log
    nvidia-smi >> ${LOG_DIR}/environment.log
    python3 -c "import torch; print(f'PyTorch版本: {torch.__version__} CUDA可用: {torch.cuda.is_available()}')" >> ${LOG_DIR}/environment.log
    python3 -c "import tensorflow as tf; print(f'TensorFlow版本: {tf.__version__} GPU可用: {len(tf.config.list_physical_devices("GPU"))>0}')" >> ${LOG_DIR}/environment.log
}

benchmark_tensorflow() {
    echo "[TensorFlow MNIST训练测试]"
    python3 -c "
import tensorflow as tf
import time

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

start_time = time.time()
model.fit(x_train, y_train, epochs=5, verbose=0)
training_time = time.time() - start_time

with open('${LOG_DIR}/tensorflow/result.log', 'w') as f:
    f.write(f'训练耗时: {training_time:.2f}秒\n')
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    f.write(f'测试准确率: {acc*100:.2f}%\n')
"
}

benchmark_pytorch() {
    echo "[PyTorch ResNet-18推理测试]"
    python3 -c "
import torch
import torchvision
import time

model = torchvision.models.resnet18(pretrained=True)
model.eval()
input = torch.rand(1, 3, 224, 224)

# Warmup
for _ in range(10):
    _ = model(input)

# 正式测试
start_time = time.time()
for _ in range(100):
    _ = model(input)
elapsed_time = time.time() - start_time

with open('${LOG_DIR}/pytorch/result.log', 'w') as f:
    f.write(f'平均推理时间: {(elapsed_time/100)*1000:.2f}毫秒\n')
"
}

main() {
    prepare_env
    benchmark_tensorflow
    benchmark_pytorch
    echo "AI框架测试完成！结果保存在：${LOG_DIR}"
}

main "$@"