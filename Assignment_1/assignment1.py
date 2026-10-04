import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def load_grayscale_image(image_path):

    image = Image.open(image_path).convert("L")

    # Convert uint8 image to float32 and normalize to [0, 1]
    image = np.array(image, dtype=np.float32) / 255.0

    return image

def generate_noisy_image(image, sigma):

    noise = np.random.normal(loc=0.0, scale=sigma, size=image.shape)

    noisy_image = image + noise

    # Keep pixel values within [0, 1]
    noisy_image = np.clip(noisy_image, 0.0, 1.0)

    return noisy_image

def average_noisy_images(original_image, K, sigma):

    noisy_images = []

    for i in range(K):

        noisy_image = generate_noisy_image(original_image, sigma)

        noisy_images.append(noisy_image)

    # Convert list into a Numpy array 
    # Shape becomes (K, height, width)

    noisy_stack = np.stack(noisy_images, axis=0)

    # Average accross the K noisy images
    average_image = np.mean(noisy_stack, axis=0)

    return average_image

def calculate_mse(original_image, processed_image):

    mse = np.mean((original_image - processed_image) ** 2)

    return mse

def run_part1():

    #--------------------------------------------------
    # Load the clean image
    #--------------------------------------------------
    image_path = "images/clean_image.png"

    original_image = load_grayscale_image(image_path)

    print("Clean image loaded successfully.")

    #--------------------------------------------------
    # Noise parameters
    #--------------------------------------------------
    variance = 0.01  # Variance of the Gaussian noise
    sigma = np.sqrt(variance)  # Standard deviation of the Gaussian noise

    print(f"Noise variance: {variance}")
    print(f"Noise standard deviation (sigma): {sigma}") 

    #--------------------------------------------------
    # Generate one noisy image
    #--------------------------------------------------
    
    noisy_image = generate_noisy_image(original_image, sigma)

    #--------------------------------------------------
    # Required K values
    #--------------------------------------------------

    K_values = [5, 10, 50, 100]

    averaged_images = []
    mse_values = []

    #--------------------------------------------------
    # Generate K noisy images, average them, and compute MSE
    #--------------------------------------------------

    for K in K_values:

        print(f"Generating {K} noisy images and averaging them...")

        average_image = average_noisy_images(original_image, K, sigma)

        mse = calculate_mse(original_image, average_image)

        averaged_images.append(average_image)
        mse_values.append(mse)

    # =====================================================
    # Display images
    # =====================================================

    plt.figure(figsize=(14, 8))

    # Original Image
    plt.subplot(2,3,1)
    plt.imshow(original_image, cmap='gray', vmin=0, vmax=1)
    plt.title('Original Image')
    plt.axis('off')
    

    # Single noisy image
    plt.subplot(2, 3, 2)
    plt.imshow(noisy_image, cmap="gray", vmin=0, vmax=1)
    plt.title("Noisy Image (K=1)")
    plt.axis("off")

    # K = 5
    plt.subplot(2, 3, 3)
    plt.imshow(
        averaged_images[0],
        cmap="gray",
        vmin=0,
        vmax=1
    )
    plt.title(f"Averaged Image (K=5)")
    plt.axis("off")

    # K = 10
    plt.subplot(2, 3, 4)
    plt.imshow(
        averaged_images[1],
        cmap="gray",
        vmin=0,
        vmax=1
    )
    plt.title(f"Averaged Image (K=10)")
    plt.axis("off")

    # K = 50
    plt.subplot(2, 3, 5)
    plt.imshow(
        averaged_images[2],
        cmap="gray",
        vmin=0,
        vmax=1
    )
    plt.title(f"Averaged Image (K=50)")
    plt.axis("off")

    # K = 100
    plt.subplot(2, 3, 6)
    plt.imshow(
        averaged_images[3],
        cmap="gray",
        vmin=0,
        vmax=1
    )
    plt.title(f"Averaged Image (K=100)")
    plt.axis("off")

    plt.suptitle(
        "Part 1: Noise Reduction by Image Averaging",
        fontsize=16
    )

    plt.tight_layout()

    # Save figure
    plt.savefig(
        "images/part1_averaging_results.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # =====================================================
    # Plot MSE vs K
    # =====================================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        K_values,
        mse_values,
        marker="o"
    )

    plt.xlabel("Number of Images (K)")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("MSE vs. Number of Averaged Images")

    plt.grid(True)

    # Use a logarithmic x-axis so that the spacing between
    # 5, 10, 50 and 100 is easier to interpret.
    plt.xscale("log")

    plt.xticks(K_values, K_values)

    plt.tight_layout()

    # Save figure
    plt.savefig(
        "images/part1_mse_plot.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # =====================================================
    # Print results table
    # =====================================================

    print("\nPart 1 Results")
    print("--------------------------------")
    print("K\tMSE")
    print("--------------------------------")

    for K, mse in zip(K_values, mse_values):

        print(f"{K}\t{mse:.8f}")

    print("--------------------------------")

    return mse_values

def main():

    # Run Part 1

    run_part1()

# Run the program
if __name__ == "__main__":
    main()