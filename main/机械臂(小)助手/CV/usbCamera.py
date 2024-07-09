import cv2

# 打开摄像头
cap = cv2.VideoCapture(2)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 设置初始焦距（假设焦距范围为0-255）
focus = 0

# 创建一个窗口
cv2.namedWindow('Camera')


def change_focus(x):
    global focus
    focus = x
    # 设置摄像头焦距
    cap.set(cv2.CAP_PROP_FOCUS, focus)


# 创建一个滑动条来调节焦距
cv2.createTrackbar('Focus', 'Camera', 0, 255, change_focus)

while True:
    # 读取摄像头帧
    ret, frame = cap.read()

    if not ret:
        print("无法接收帧 (stream end?). Exiting ...")
        break

    # 显示当前帧
    cv2.imshow('Camera', frame)

    # 按下'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭窗口
cap.release()
cv2.destroyAllWindows()
