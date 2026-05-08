import matplotlib.pyplot as plt

epochs = list(range(1, 21))

train_loss = [0.3057, 0.1397, 0.1106, 0.1038, 0.0913, 0.0803, 0.0703, 0.0715,
              0.0606, 0.0639, 0.0550, 0.0499, 0.0477, 0.0497, 0.0409, 0.0396,
              0.0377, 0.0377, 0.0345, 0.0310]

val_loss =   [0.1562, 0.1145, 0.1196, 0.0991, 0.0926, 0.0933, 0.1076, 0.1052,
              0.0961, 0.1045, 0.1005, 0.1074, 0.1097, 0.1318, 0.1654, 0.1111,
              0.1105, 0.1108, 0.1202, 0.1436]

train_acc =  [0.9003, 0.9545, 0.9619, 0.9647, 0.9671, 0.9705, 0.9762, 0.9750,
              0.9795, 0.9775, 0.9791, 0.9824, 0.9834, 0.9826, 0.9862, 0.9859,
              0.9864, 0.9866, 0.9872, 0.9895]

val_acc =    [0.9432, 0.9564, 0.9579, 0.9661, 0.9661, 0.9696, 0.9636, 0.9657,
              0.9711, 0.9700, 0.9629, 0.9675, 0.9657, 0.9636, 0.9546, 0.9704,
              0.9682, 0.9714, 0.9668, 0.9604]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(epochs, train_loss, label="Train Loss")
ax1.plot(epochs, val_loss, label="Val Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training vs Validation Loss")
ax1.legend()
ax1.grid(True)
ax1.set_xticks(epochs)

ax2.plot(epochs, train_acc, label="Train Accuracy")
ax2.plot(epochs, val_acc, label="Val Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Training vs Validation Accuracy")
ax2.legend()
ax2.grid(True)
ax2.set_xticks(epochs)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()