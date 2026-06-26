from main_python import run_pure_python
from main_numpy import run_numpy
from main_skl import run_sklearn
import matplotlib.pyplot as plt

print("____PYTHON____")
python = run_pure_python()
print("____NUMPY____")
numpy = run_numpy()
print("____SKLEARN____")
sklearn = run_sklearn()

#Cost vs iterations
plt.figure()
plt.plot(range(0, len(python["costs"]) * 20, 20),
         python["costs"],
         label="Pure Python")

plt.plot(range(0, len(numpy["costs"]) * 20, 20),
         numpy["costs"],
         label="NumPy")

plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.legend()

# Cost vs Time
plt.figure()
plt.plot(python["times"],
         python["costs"],
         label="Pure Python")

plt.plot(numpy["times"],
         numpy["costs"],
         label="NumPy")

plt.xlabel("Time (seconds)")
plt.ylabel("Cost")
plt.legend()

#Training Timee
plt.figure()
labels = ["Pure Python", "NumPy", "Scikit-learn"]
times = [
    python["training_time"],
    numpy["training_time"],
    sklearn["training_time"]
]

plt.bar(labels, times)
plt.ylabel("Training Time (seconds)")

#R2 Score 
plt.figure()
labels=["Pure Python", "NumPy", "Scikit-learn"]
r2 = [
    python["r2"],
    numpy["r2"],
    sklearn["r2"]
]

plt.bar(labels, r2)
plt.ylabel("R² Score")

#Mean Absolute Error 
plt.figure()
labels=["Pure Python", "NumPy", "Scikit-learn"]
mae = [
    python["mae"],
    numpy["mae"],
    sklearn["mae"]
]

plt.bar(labels, mae)
plt.ylabel("Mean Absolute Error")

# #Root Mean Squared Error 
plt.figure()
labels=["Pure Python", "NumPy", "Scikit-learn"]
rmse = [
    python["rmse"],
    numpy["rmse"],
    sklearn["rmse"]
]

plt.bar(labels, rmse)
plt.ylabel("Root Mean Squared Error")


plt.show()

