import matplotlib.pyplot as plt
import torch

if __name__ == "__main__":
    checkpoint = "./checkpoint-big-100/final.pt"
    data = torch.load(checkpoint)
    losses = data["losses"]

    plt.figure(figsize=(10, 6))
    plt.title("Training Loss")
    plt.semilogy(range(len(losses)), losses)
    plt.xlabel("Iteration")
    plt.ylabel("Log Loss")

    plt.savefig("./training_loss.png")
