import numpy as np

binary_file = '/Users/kitlewers/Desktop/naive_compression/imagery/ang20231109t092617_027_L2A_OE_main_27577724_RFL_ORT'  # Replace with your binary file path
hdr_file = '/Users/kitlewers/Desktop/naive_compression/imagery/ang20231109t092617_027_L2A_OE_main_27577724_RFL_ORT.hdr'  # Replace with your header file path

data = load_binary_file(binary_file, hdr_metadata)
print("Minimum value:", np.min(data))
print("Maximum value:", np.max(data))
