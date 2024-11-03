import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import xbitinfo as xb
from xbitinfo import get_bitinformation, get_keepbits

def load_hyperspectral_data(file_path, header_path):
    # Load header information
    with open(header_path, 'r') as f:
        header = f.readlines()
    # Extract metadata from header
    samples = int([line.split('=')[1] for line in header if 'samples' in line][0])
    lines = int([line.split('=')[1] for line in header if 'lines' in line][0])
    bands = int([line.split('=')[1] for line in header if 'bands' in line][0])
    dtype = np.float32  # Assuming data type is 4 (float32 in ENVI standard)
    
    # Load binary data
    data = np.fromfile(file_path, dtype=dtype)
    data = data.reshape((bands, lines, samples))
    da = xr.DataArray(data, dims=("band", "y", "x"))
    return da

def calculate_bit_information(da):
    # Wrap DataArray in a Dataset
    ds = da.to_dataset(name="data")
    return get_bitinformation(ds, dim="band")

def find_optimal_bits(bit_info, info_level=0.99):
    # Get keep_bits from xbitinfo
    keep_bits = get_keepbits(bit_info, info_level)
    
    # Convert to scalar if keep_bits is an array or DataArray
    if isinstance(keep_bits, xr.Dataset):
        keep_bits = keep_bits.to_array().values  # Convert Dataset to numpy array
    if isinstance(keep_bits, xr.DataArray):
        keep_bits = keep_bits.values  # Convert DataArray to numpy array
    if isinstance(keep_bits, np.ndarray):
        keep_bits = keep_bits.item()  # Convert numpy array to scalar if it's a single value
    
    return keep_bits


def plot_bit_information(bit_info, keep_bits):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the information content for each bit in each band
    for band in bit_info.data_vars:
        ax.plot(bit_info[band].values, label=f'Band {band}')
        
    # Highlight the bit depth needed to retain 99% of information
    ax.axvline(keep_bits, color='red', linestyle='--', label=f'99% Retention ({keep_bits} bits)')
    ax.set_title("Bitwise Information Content per Band")
    ax.set_xlabel("Bit Position")
    ax.set_ylabel("Information Content")
    ax.legend()
    plt.show()

def add_bitinfo_labels(da, info_per_bit, inflevels=None, keepbits=None, ax=None, x_dim_name="x", y_dim_name="y"):
    if inflevels is None and keepbits is None:
        raise KeyError("Either inflevels or keepbits need to be provided")
    if ax is None:
        ax = plt.gca()

    stride = da[x_dim_name].size // len(inflevels)
    for i, inf in enumerate(inflevels):
        lons = da.isel({x_dim_name: stride * i})
        lats = da.isel({x_dim_name: stride * i})
        lons, lats = xr.broadcast(lons, lats)
        ax.plot(lons, lats, color="k", linewidth=1)

        t = ax.text(
            da.isel(
                {
                    x_dim_name: int(stride * (i + 0.5)),
                    y_dim_name: da[y_dim_name].size // 2,
                }
            )[x_dim_name].values,
            lats.mean().values,
            f"{round(inf * 100, 2)}%",
            horizontalalignment="center",
            color="k",
        )
        t.set_bbox(dict(facecolor="white", alpha=0.9, edgecolor="white"))

def main():
    binary_file_path = "/Users/kitlewers/Desktop/naive_compression/imagery/ang20231109t092617_027_L2A_OE_main_27577724_RFL_ORT"
    header_file_path = "/Users/kitlewers/Desktop/naive_compression/imagery/ang20231109t092617_027_L2A_OE_main_27577724_RFL_ORT.hdr"
    
    # Load hyperspectral data
    da = load_hyperspectral_data(binary_file_path, header_file_path)
    
    # Calculate bitwise information content
    bit_info = calculate_bit_information(da)
    
    # Determine optimal bit depth for 99% information retention
    keep_bits = find_optimal_bits(bit_info, info_level=0.99)
    print(f"Bits required for 99% retention: {keep_bits}")
    
    # Plot bitwise information content
    plot_bit_information(bit_info, keep_bits)
    
    # Add bit info labels to visualize bit information with labels
    fig, ax = plt.subplots()
    inflevels = [0.95, 0.99]
    add_bitinfo_labels(da, bit_info, inflevels=inflevels, keepbits=keep_bits, ax=ax)
    plt.show()

# Execute the main function
if __name__ == "__main__":
    main()
