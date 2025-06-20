import tensorflow as tf

# GPU 목록 출력
gpus = tf.config.list_physical_devices('GPU')
print("Available GPUs:", gpus)

# GPU가 있다면 연산을 GPU에서 수행
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)  # GPU 메모리 제한 충돌 방지
        with tf.device('/GPU:0'):
            a = tf.random.normal((1000, 1000))
            b = tf.random.normal((1000, 1000))
            c = tf.matmul(a, b)
        print("GPU computation succeeded:", c.numpy()[0, 0])
    except Exception as e:
        print("GPU exists but computation failed:", e)
else:
    print("No GPU available.")
