import matplotlib.pyplot as plt


class ModelHistoryPlotter:
    def __init__(self, model):
        self.history = model.history.history

    def plot_history(self):
        fig, axes = plt.subplots(1, 2, figsize=(20, 6))
        axes[0].plot(self.history['accuracy'], label="Training Accuracy")
        # axes[0].plot(self.history['val_accuracy'], label="Validation Accuracy")
        axes[0].set_title("Accuracy Plot")
        axes[0].set_xlabel("Epochs")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()

        axes[1].plot(self.history['loss'], label="Training Loss")
        # axes[1].plot(self.history['val_loss'], label="Validation Loss")
        axes[1].set_title("Loss Plot")
        axes[1].set_xlabel("Epochs")
        axes[1].set_ylabel("Loss")
        axes[1].legend()

        plt.tight_layout()
        return fig
