# === FILE: simulation/plot_results.py ===
import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob

def plot_logical_errors():
    latest_csv = sorted(glob("results/simulations_*.csv"))[-1]
    df = pd.read_csv(latest_csv)

    plt.figure(figsize=(8, 6))
    for code in df['code'].unique():
        subset = df[(df['code'] == code) & (df['noise_type'] == "depolarizing")]
        grouped = subset.groupby('noise_prob')['logical_error'].mean()
        plt.plot(grouped.index, grouped.values, label=code, marker='o')

    plt.xlabel("Noise Probability")
    plt.ylabel("Logical Error Rate")
    plt.title("Logical Error Rate vs Noise Probability")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/logical_error_plot.png")
    plt.show()

def plot_decoder_performance():
    latest_decoder_csv = sorted(glob("results/decoder_performance_*.csv"))[-1]
    df = pd.read_csv(latest_decoder_csv)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.bar(df["Code"], df["Decoder Time (s)"])
    plt.title("Decoder Runtime (s)")
    plt.ylabel("Time (s)")

    plt.subplot(1, 2, 2)
    plt.bar(df["Code"], df["Memory Usage (KB)"])
    plt.title("Decoder Memory Usage (KB)")
    plt.ylabel("Memory (KB)")

    plt.tight_layout()
    plt.savefig("results/decoder_complexity.png")
    plt.show()
