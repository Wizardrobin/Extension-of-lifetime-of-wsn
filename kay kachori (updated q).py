import simpy
import numpy as np
import random
import turtle
import time
import heapq

# Set fixed random seed for reproducibility
random.seed(50)
np.random.seed(50)

# Function to get valid integer input
def get_valid_input(prompt, min_value, max_value):
    while True:
        try:
            value = int(input(prompt))
            if min_value <= value <= max_value:
                return value
            else:
                print(f"Please enter a number between {min_value} and {max_value}.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

# Parameters
num_nodes = get_valid_input("Enter the number of nodes: ", 1, 100)
episodes = 100
learning_rate = 0.5
discount_factor = 0.8
exploration_rate = 1.0
exploration_decay = 0.995
min_exploration_rate = 0.01
energy_threshold = 0

# Assign a random energy level to each node (between 15 and 100)
energy_levels = np.random.randint(15, 100, size=num_nodes)

# Get initial and final states
initial_state = get_valid_input(f"Enter the initial node (0 to {num_nodes - 1}): ", 0, num_nodes - 1)
final_state = get_valid_input(f"Enter the final node (0 to {num_nodes - 1}): ", 0, num_nodes - 1)

# Initialize Q-table with connectivity
q_table = np.random.rand(num_nodes, num_nodes) * 10  
np.fill_diagonal(q_table, 0)  # No self-loops

# Print all Q-learning weights
print("\nQ-learning Weights:")
for i in range(num_nodes):
    for j in range(num_nodes):
        print(f"Q[{i}][{j}] = {q_table[i][j]:.2f}", end="  ")
    print()

# Turtle setup
screen = turtle.Screen()
screen.setup(950, 800)
screen.title("Node Simulation")
screen.tracer(0)

# Node positions
positions = {i: (random.randint(-350, 350), random.randint(-350, 350)) for i in range(num_nodes)}

# Create turtles for nodes
node_turtles = {}
for i in range(num_nodes):
    t = turtle.Turtle()
    t.penup()
    t.shape("circle")
    t.color("green")
    t.goto(positions[i])
    t.write(f"Node {i}\nEnergy: {energy_levels[i]}", align="center", font=("Arial", 10, "normal"))
    node_turtles[i] = t

# Draw all connections
def draw_all_connections():
    turtle.pensize(1)
    turtle.pencolor("lightgray")
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                turtle.penup()
                turtle.goto(positions[i])
                turtle.pendown()
                turtle.goto(positions[j])
    screen.update()

draw_all_connections()

# Draw optimal path

def draw_optimal_path(path):
    if not path:
        print("No valid path found!")
        return
    
    turtle.penup()
    turtle.pensize(3)
    turtle.pencolor("red")
    for i in range(len(path) - 1):
        turtle.goto(positions[path[i]])
        turtle.pendown()
        turtle.goto(positions[path[i + 1]])
    
    for node in path:
        node_turtles[node].color("red")
    screen.update()

# Dijkstra’s Algorithm for optimal path
def find_optimal_path(start_node, end_node):
    distances = {node: float('inf') for node in range(num_nodes)}
    previous_nodes = {node: None for node in range(num_nodes)}
    distances[start_node] = 0
    pq = [(0, start_node)]

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        if current_node == end_node:
            break

        for neighbor in range(num_nodes):
            if neighbor == current_node or energy_levels[neighbor] <= energy_threshold:
                continue
            
            weight = q_table[current_node][neighbor]
            if weight > 0:
                distance = current_distance + weight
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    previous_nodes[neighbor] = current_node
                    heapq.heappush(pq, (distance, neighbor))
    
    path = []
    current = end_node
    while current is not None:
        path.append(current)
        current = previous_nodes[current]
    
    path.reverse()
    return path if path and path[0] == start_node else []

# Initialize simulation
env = simpy.Environment()
optimal_path = find_optimal_path(initial_state, final_state)
draw_optimal_path(optimal_path)

print("\nOptimal Path:", " -> ".join(map(str, optimal_path)) if optimal_path else "No valid path found.")

screen.mainloop()
