from food_truck_env import FoodTruck

ft = FoodTruck()
print(ft.possible_actions([7,1]))
possible_future_states = [[7,1] + ft.get_change(act) for act in ft.possible_actions([7, 1])]
print(possible_future_states)
print(ft.ft_map)