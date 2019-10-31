from food_truck import FoodTruck
import torch

ft = FoodTruck()
print(ft.R.shape)
print(torch.argmax(ft.R))