import csv
import sys
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print("Usage: python plot_learning_curve.py your_learning_curve.csv")
    sys.exit(1)

csv_path = sys.argv[1]

steps = []
rewards = []

with open(csv_path, "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        steps.append(float(row["num_steps"]))
        rewards.append(float(row["eval_episode_reward"]))

plt.figure()
plt.plot(steps, rewards)
plt.xlabel("Training steps")
plt.ylabel("Eval episode reward")
plt.title("Learning curve")
plt.grid(True)
plt.show()