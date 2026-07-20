import matplotlib.pyplot as plt
import matplotlib.image as mpimg

fig, axes = plt.subplots(3, 1, figsize=(10, 18))

baseline_img = mpimg.imread("output/baseline_training_curves.png")
axes[0].imshow(baseline_img)
axes[0].axis("off")
axes[0].set_title("Baseline Model - Loss and Accuracy over Epochs", fontsize=14)

optimized_img = mpimg.imread("output/optimized_training_curves.png")
axes[1].imshow(optimized_img)
axes[1].axis("off")
axes[1].set_title("Optimized Model - Loss and Accuracy over Epochs", fontsize=14)

confusion_img = mpimg.imread("output/confusion_matrix.png")
axes[2].imshow(confusion_img)
axes[2].axis("off")
axes[2].set_title("Confusion Matrix - Optimized Model (Test Set)", fontsize=14)

plt.tight_layout()
plt.savefig("output/final_results_grid.png", dpi=150, bbox_inches="tight")
print("Combined grid saved to output/final_results_grid.png")