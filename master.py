#Python - Sudoku Solver with Deep Learning Digit Recognizer
""" 
This program will take an input of a Sudoku puzzle and then use a deep learning algorithm to recognize the digits.
Afterwards it will solve the puzzle using _______ algorithm and then output that to an array.
The program will finally take this array and output it to a .jpg as a solved puzzle.
"""

import PIL
from PIL import Image
import tkinter as tk
from tkinter import filedialog
from torchvision import models, transforms
import torch

print('Pillow Version: ', PIL.__version__)
root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()
sudoku = Image.open(file_path)


print(sudoku.format)
print(sudoku.mode)
print(sudoku.size)

#sudoku.show()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # THESE VALUES ARE GIVEN BY TORCHVISION DOCS
        std=[0.229, 0.224, 0.225]  # IN ORDER TO USE PRE-TRAINED MODELS
    )
])

sudoku_torch = transform(sudoku)
batch_t = torch.unsqueeze(sudoku_torch, 0)

print(sudoku_torch)

resnet18 = models.resnet18(pretrained=True)
resnet18.eval()

out = resnet18(batch_t)

print(out)

imgout = transforms.ToPILImage()(sudoku_torch)

imgout.show()

# sudoku.save('sudoku_puzzle_solved.jpg', format='JPG')



# read file in as .jpg
# generate numbers using a pre-config'd algorithm w/tuning
# solve
# output to arrays
# change array to image file

print('test')
