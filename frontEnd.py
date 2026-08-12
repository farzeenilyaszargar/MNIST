import tkinter as tk
import numpy as np
from PIL import Image, ImageDraw

# load model
m = np.load("mnist_model.npz")
W1, b1 = m["weights1"], m["biases1"]
W2, b2 = m["weights2"], m["biases2"]

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

def predict():
    img = drawing.resize((28, 28))
    x = np.array(img, dtype=np.float32).reshape(784) / 255.0

    a1 = np.maximum(0, x @ W1 + b1)
    prob = softmax(a1 @ W2 + b2)

    result.config(text=f"Prediction: {np.argmax(prob)}")
    probs.config(text="\n".join(
        f"{i}: {prob[i]*100:.1f}%" for i in range(10)
    ))

def draw(event):
    x, y = event.x, event.y
    canvas.create_oval(x-10, y-10, x+10, y+10, fill="white", outline="white")
    pen.ellipse((x-10, y-10, x+10, y+10), fill=255)

def clear():
    global drawing, pen
    canvas.delete("all")
    drawing = Image.new("L", (280, 280), 0)
    pen = ImageDraw.Draw(drawing)
    result.config(text="Prediction: ?")
    probs.config(text="")

root = tk.Tk()
root.title("MNIST")


drawing = Image.new("L", (280, 280), 0)
pen = ImageDraw.Draw(drawing)

canvas = tk.Canvas(root, width=280, height=280, bg="black")
canvas.pack()
canvas.bind("<B1-Motion>", draw)

tk.Button(root, text="Predict", command=predict).pack()
tk.Button(root, text="Clear", command=clear).pack()

result = tk.Label(root, text="Prediction: ?", font=("Arial", 20))
result.pack()

probs = tk.Label(root)
probs.pack()

root.mainloop()