# Using KNN b/c Fischer Geodisic min distance soudned similar
# Procedural based SA

import random
import matplotlib.pyplot as plt

list_of_colors = ["blue", "black", "green", "purple", "pink"]

# Create a random dataset
def create_dataset(len_dataset):
    dataset = []
    placeholder = []
    
    for i in range(len_dataset):
                
        placeholder.append(random.randint(1, 100))
        placeholder.append(random.randint(1, 100))
        
        dataset.append(placeholder)
        placeholder = []
                    
    return (dataset)

user_input1 = int(input("# of points: "))

my_dataset = create_dataset(user_input1)

# Create a random point based on lower and upper bounds
def create_point(lower_bound, upper_bound):
    test_point = ['x', 'y']
    
    test_point[0] = random.randint(lower_bound, upper_bound)
    test_point[1] = random.randint(lower_bound, upper_bound)
    
    return test_point
    
new_point = create_point(1, 100)

new_point = [new_point[0], new_point[1]]

# Find the closest distance 
def find_closest_distance(dataset, new_point):
    list_of_distances = []
    
    for i in range(len(dataset)):
        
        x_distance = abs(dataset[i][0] - new_point[0])
        y_distance = abs(dataset[i][1] - new_point[1])
        
        total_distance = (x_distance**2 + y_distance**2) ** 0.5
        
        list_of_distances.append(total_distance)
        
    return list_of_distances

# Add random colors
# These classes are not meant to mimic real-life data (classes shouldn't be so scattered)
def add_colors(dataset, color_list):
    for i in range(len(dataset)):
        random_color = color_list[random.randint(0, len(color_list)-1)]
        dataset[i].append(random_color)
    return dataset

def find_colour(some_dataset, list_of_distances, n_value):
    temp_dataset = []

    for i in some_dataset:
        temp_dataset.append(i)

    colors_list = []

    for i in range(n_value):
        min_distance = min(list_of_distances)
        index = my_list_of_distances.index(min_distance)

        classify_color = temp_dataset[index][2]

        list_of_distances.pop(index)
        temp_dataset.pop(index)

        colors_list.append(classify_color)

    color_count = []

    for i in range(n_value):
        count = colors_list.count(colors_list[i])
        color_count.append(count) 

    max_color = max(color_count)
    index_color = color_count.index(max_color)

    print(colors_list)

    return colors_list[index_color]

        
my_dataset = add_colors(my_dataset, list_of_colors)

my_list_of_distances = find_closest_distance(my_dataset, new_point)

neighbors_value = int(input("How many neighbours are counted?: "))

classify_color = find_colour(my_dataset, my_list_of_distances, neighbors_value)

for i in range(len(my_dataset)):
    plt.scatter(my_dataset[i][0], my_dataset[i][1], color=my_dataset[i][2])

print(new_point)
print(classify_color)
    
plt.scatter(new_point[0],new_point[1], color="red")
plt.title("KNN example")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.show()