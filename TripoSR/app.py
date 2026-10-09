import argparse
import tkinter as tk
from PIL import Image, ImageTk

parser = argparse.ArgumentParser()
parser.add_argument("image", type=str, nargs="+", help="Path to input image(s).")

args = parser.parse_args()

root = tk.Tk()
root.title("Image Preview")


# Load and display an image 
image = Image.open(args.image[0])
print(image.size)
image = image.resize((256, 256))
image = ImageTk.PhotoImage(image)

image_label = tk.Label(root, image=image)
image_label.grid(row=1, column=0)

image2 = Image.open("examples/car.png")
image2 = image2.resize((256, 256))
image2 = ImageTk.PhotoImage(image2)

image_label2 = tk.Label(root, image=image2)
image_label2.grid(row=1, column=1)

text_label = tk.Label(root, text="Input Image")
text_label.grid(row=0, column=0)

root.mainloop()