import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os


label_inputs = [""] # Put ur model names here
OUTPUT_DIR = Path("graphs")

def main(label_input):
    # Load CSV
    df = pd.read_csv(f"models/{label_input}/loss.csv")  # change filename if needed

    # Metrics to plot
    metrics = ["Alpha_Loss", "Critic_Loss", "Actor_Loss", "Alpha"]

    # -------- Plot 1: Metrics vs Global_Steps --------
    fig, axs = plt.subplots(len(metrics), 1, figsize=(10, 12), sharex=True)
    for i, metric in enumerate(metrics):
        axs[i].plot(df["Global_Steps"], df[metric], label=metric)
        axs[i].set_ylabel(metric)
        axs[i].legend()
    axs[-1].set_xlabel("Global Steps")
    plt.tight_layout()
    os.makedirs(OUTPUT_DIR / label_input, exist_ok=True)
    plt.savefig(OUTPUT_DIR / label_input / "loss_vs_global_steps.png")
    plt.close()

    # -------- Plot 2: Metrics vs Episode --------
    fig, axs = plt.subplots(len(metrics), 1, figsize=(10, 12), sharex=True)
    for i, metric in enumerate(metrics):
        axs[i].plot(df["Episode"], df[metric], label=metric)
        axs[i].set_ylabel(metric)
        axs[i].legend()
    axs[-1].set_xlabel("Episode")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / label_input / "loss_vs_episode.png")
    plt.close()

    print("Saved plots: metrics_vs_global_steps.png and metrics_vs_episode.png")

    df = pd.read_csv(f"models/{label_input}/train.csv") # replace with your CSV path

    # Assuming your CSV has columns: 'Global_Steps' and 'Episode' and 'Reward'
    # If 'Reward' isn't directly given, you might sum or compute it per episode

    # Example: if your CSV has total reward per episode in 'Total_Reward' column
    plt.figure(figsize=(8,5))
    plt.plot(df['GlobalSteps'], df['TotalReward'])
    plt.xlabel("Global Steps")
    plt.ylabel("Total Reward")
    plt.title("Total Rewards vs Global Steps")
    plt.grid(True)

    # Save the figure
    plt.savefig(OUTPUT_DIR / label_input / "rewards_vs_steps.png", dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure so it doesn't display

    df = pd.read_csv(f"models/{label_input}/val.csv") # replace with your CSV path

    # Assuming your CSV has columns: 'Global_Steps' and 'Episode' and 'Reward'
    # If 'Reward' isn't directly given, you might sum or compute it per episode

    # Example: if your CSV has total reward per episode in 'Total_Reward' column
    # breakpoint()
    plt.figure(figsize=(8,5))
    plt.plot(df['GlobalStep'], df['AverageReward'])
    plt.xlabel("Global_Step")
    plt.ylabel("AverageReward")
    plt.title("AverageReward vs Global Steps")
    plt.grid(True)

    # Save the figure
    plt.savefig(OUTPUT_DIR / label_input / "eval_rewards_vs_steps.png", dpi=300, bbox_inches='tight')
    plt.close()  # Close the figure so it doesn't display


for i in label_inputs:
    try:
        main(i)
    except:
        print("failed ", i)