import cv2
import os

# 创建测试目录
os.makedirs('camera_test', exist_ok=True)

# 测试所有video节点
for i in range(15):
    dev = f'/dev/video{i}'
    cap = cv2.VideoCapture(dev)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            filename = f'camera_test/video{i}_{w}x{h}.jpg'
            cv2.imwrite(filename, frame)
            print(f'✅ video{i}: {w}x{h} -> 保存到 {filename}')
        else:
            print(f'⚠️ video{i}: 能打开但无法读取帧')
        cap.release()
    else:
        print(f'❌ video{i}: 无法打开')