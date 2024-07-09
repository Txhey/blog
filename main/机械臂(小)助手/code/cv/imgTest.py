import tkinter as tk
from PIL import Image, ImageTk

class ImageBlenderApp:
    def __init__(self, master, image1_path, image2_path):
        self.master = master
        self.master.title("Image Blender")

        # 打开并处理两张图片
        self.image1 = Image.open(image1_path).convert('RGBA')
        self.image2 = Image.open(image2_path).convert('RGBA')
        self.width, self.height = self.image1.size

        # 设置两张图片的透明度为50%
        self.image1 = self.set_image_alpha(self.image1, 0.5)
        self.image2 = self.set_image_alpha(self.image2, 0.5)

        # 初始化合成后的图片
        self.blended_image = Image.alpha_composite(self.image1, self.image2)

        # 创建Tkinter画布
        self.canvas = tk.Canvas(self.master, width=self.width, height=self.height)
        self.canvas.pack()

        # 显示初始合成图片
        self.update_display()

        # 绑定按键事件
        self.master.bind("<KeyPress>", self.on_key_press)

    def set_image_alpha(self, image, alpha):
        # 分离通道
        r, g, b, a = image.split()
        # 调整透明度
        a = a.point(lambda i: i * alpha)
        return Image.merge('RGBA', (r, g, b, a))

    def update_display(self):
        # 将PIL Image转换为Tkinter PhotoImage
        self.photo = ImageTk.PhotoImage(self.blended_image)
        # 更新画布上的图片
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

    def on_key_press(self, event):
        if event.keysym == 'Right':
            # 向右移动图片2
            self.move_image2(1)
            # 更新合成图片
            self.blended_image = Image.alpha_composite(self.image1, self.image2)
            # 更新显示
            self.update_display()
        if event.keysym == 'Left':
            # 向右移动图片2
            self.move_image2(-1)
            # 更新合成图片
            self.blended_image = Image.alpha_composite(self.image1, self.image2)
            # 更新显示
            self.update_display()

    def move_image2(self, pixels):
        # 计算新的位置
        new_position = (self.image2.size[0] - pixels, 0)
        # 创建新的图片对象
        new_image2 = Image.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        new_image2.paste(self.image2, new_position)
        self.image2 = new_image2

if __name__ == "__main__":
    root = tk.Tk()
    image1_path = "img/c1.png"
    image2_path = "img/c2.png"

    app = ImageBlenderApp(root, image1_path, image2_path)

    root.mainloop()
