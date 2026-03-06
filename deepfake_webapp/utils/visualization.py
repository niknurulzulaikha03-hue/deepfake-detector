import matplotlib.pyplot as plt
import numpy as np

def generate_probability_chart(real_prob, fake_prob, output_path):

    labels = ['Real', 'Fake']
    values = [real_prob, fake_prob]

    plt.figure()

    plt.bar(labels, values)

    plt.title("Deepfake Prediction Probability")

    plt.savefig(output_path)

    plt.close()


def generate_mfcc_plot(mfcc, output_path):

    plt.figure()

    plt.imshow(mfcc, aspect='auto', origin='lower')

    plt.title("MFCC Spectrogram")

    plt.colorbar()

    plt.savefig(output_path)

    plt.close()
