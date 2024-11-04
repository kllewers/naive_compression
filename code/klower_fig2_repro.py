import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xbitinfo as xb
from xbitinfo import get_bitinformation

def load_hyperspectral_data(file_path, header_path):
    with open(header_path, 'r') as f:
        header = f.readlines()
    
    # Extract metadata from header
    samples = int([line.split('=')[1] for line in header if 'samples' in line][0])
    lines = int([line.split('=')[1] for line in header if 'lines' in line][0])
    bands = int([line.split('=')[1] for line in header if 'bands' in line][0])
    dtype = np.float32  # Assuming data type is float32
    
    # Load binary data and mask -9999 values
    data = np.fromfile(file_path, dtype=dtype).reshape((bands, lines, samples))
    data = np.where(data == -9999, np.nan, data)  # Mask invalid values
    
    # Convert to xarray DataArray
    da = xr.DataArray(data, dims=("band", "y", "x"))
    return da

def plot_bit_information_figure2(data_array):
    # Convert DataArray to Dataset if needed and calculate bit information
    dataset = data_array.to_dataset(name="data")
    bit_info = get_bitinformation(dataset, dim="band")
    
    # Plot bitwise information content using `plot_bitinformation`
    xb.plot_bitinformation(bit_info, cmap="turku", crop=64)  # Adjust `crop` if fewer bits are relevant
    plt.show()

def main():
    binary_file_path = "/Users/kitlewers/Desktop/naive_compression/imagery/ang20231109t092617_027_L2A_OE_main_27577724_RFL_ORT"
    header_file_path = "/Users/kitlewers/Desktop/naive_compression/imagery/ang20231109t092617_027_L2A_OE_main_27577724_RFL_ORT.hdr"
    
    # Load hyperspectral data
    da = load_hyperspectral_data(binary_file_path, header_file_path)
    
    # Plot bitwise information content as in Figure 2
    plot_bit_information_figure2(da)

# Run the main function
if __name__ == "__main__":
    main()
