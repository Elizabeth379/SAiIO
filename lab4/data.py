import random

num_items: int = 4
max_weight: int = 15
max_value: int = 20
capacity: int = 15

weights: list = [random.randint(1, max_weight) for _ in range(num_items)]
values: list = [random.randint(1, max_value) for _ in range(num_items)]