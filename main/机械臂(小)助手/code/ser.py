import serial

# 串口配置
ser = serial.Serial('COM6', 1500000, timeout=1)  # 修改为你的串口设备名称

print("开始接收数据...")

try:
    while True:
        if ser.in_waiting > 0:
            message = ser.readline().decode().strip()
            print(f"接收: {message}")
except KeyboardInterrupt:
    print("\n程序终止")
    ser.close()
