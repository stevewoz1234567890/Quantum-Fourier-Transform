import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit.library import QFT
from PIL import Image
import matplotlib.pyplot as plt

def image_to_quantum_state(image_data):
    vector = image_data.flatten() / 255.0
    norm = np.linalg.norm(vector)
    if norm == 0:
        norm = 1
    normalized_vector = vector / norm
    return normalized_vector

def get_key(image_data):
    max_, min_ = max(image_data.flatten()), min(image_data.flatten())
    return max_*1000 + min_

def quantum_process_image(image_data, key_phase, encrypt=True):
    num_qubits = int(np.log2(len(image_data.flatten())))
    circuit = QuantumCircuit(num_qubits)

    vector = image_to_quantum_state(image_data)
    circuit.initialize(vector.tolist(), range(num_qubits))

    circuit.append(QFT(num_qubits), range(num_qubits))

    phase, key = key_phase % 10, key_phase // 10
    max_, min_ = key // 1000, key_phase % 1000

    phase_modifier = phase if encrypt else -phase
    for i in range(num_qubits):
        circuit.p(phase_modifier*i, i)

    circuit.append(QFT(num_qubits, inverse=True), range(num_qubits))

    simulator = Aer.get_backend('statevector_simulator')
    result = execute(circuit, simulator).result()
    statevector = result.get_statevector()

    processed_image_data = np.abs(statevector).reshape(image_data.shape)
    processed_image_data = (((processed_image_data-np.min(processed_image_data)) / (np.max(processed_image_data)-np.min(processed_image_data)) * (max_-min_)+min_)*1).astype(np.uint8)
    return processed_image_data

def calculate_mse(image1, image2):
    return np.mean((image1 - image2) ** 2)

image_path = 'D:/work/QFT/0000.png'
with Image.open(image_path) as img:
    img = img.convert("L")
    img = img.resize((128, 128))  # Smaller size for demonstration
    original_image_data = np.array(img, dtype=float)

key = get_key(original_image_data)
# key_phases = np.linspace(0, np.pi, 1000)
key_phases = [0] #[np.pi/10000000000]
mse_values = []

for phase in key_phases:
    print(phase)
    encrypted_image_data = quantum_process_image(original_image_data, key*10 + phase, encrypt=True)
    decrypted_image_data = quantum_process_image(encrypted_image_data, key*10 + phase, encrypt=False)
    mse = calculate_mse(original_image_data, decrypted_image_data)
    print(mse)
    mse_values.append(mse)

optimal_phase = key_phases[np.argmin(mse_values)]
optimal_mse = min(mse_values)

# Process images with the optimal key phase
optimal_encrypted_image = quantum_process_image(original_image_data, key*10 + optimal_phase, encrypt=True)
optimal_decrypted_image = quantum_process_image(optimal_encrypted_image, key*10 + optimal_phase, encrypt=False)
# optimal_encrypted_image = quantum_process_image(original_image_data, key, encrypt=True)
# optimal_decrypted_image = quantum_process_image(optimal_encrypted_image, key, encrypt=False)

# Display the results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.imshow(original_image_data, cmap='gray', vmin=0, vmax=255)
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(optimal_encrypted_image, cmap='gray', vmin=0, vmax=255)
plt.title('Encrypted Image')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(optimal_decrypted_image, cmap='gray', vmin=0, vmax=255)
plt.title('Decrypted Image')
plt.axis('off')
plt.suptitle(f'Optimal Key Phase: {optimal_phase:.2f} radians, MSE: {optimal_mse:.2f}')
plt.show()

# print(min(original_image_data.flatten()), max(original_image_data.flatten()))
# print(min(optimal_decrypted_image.flatten()), max(optimal_decrypted_image.flatten()))


def calculate_metrics(original, encrypted):
    original = original.astype(np.float32)
    encrypted = encrypted.astype(np.float32)
    ssim_value = ssim(original, encrypted, data_range=encrypted.max() - encrypted.min())
    mse = np.mean((original - encrypted) ** 2)
    if mse == 0:
        psnr_value = 100
    else:
        PIXEL_MAX = 255.0
        psnr_value = 20 * np.log10(PIXEL_MAX / np.sqrt(mse))
    return ssim_value, psnr_value

from skimage.metrics import structural_similarity as ssim

# Calculate metrics
ssim_enc, psnr_enc = calculate_metrics(original_image_data, encrypted_image_data)
ssim_dec, psnr_dec = calculate_metrics(original_image_data, decrypted_image_data)

# Output results
print(f"Encryption - SSIM: {ssim_enc}, PSNR: {psnr_enc}")
print(f"Decryption - SSIM: {ssim_dec}, PSNR: {psnr_dec}")