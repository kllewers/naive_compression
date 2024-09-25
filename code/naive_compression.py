import numpy as np
import xarray as xr
import os

def parse_hdr_file(hdr_file):
    """Parse the ENVI .hdr file to extract metadata."""
    metadata = {}
    with open(hdr_file, 'r') as file:
        for line in file:
            key_value = line.strip().split('=')
            if len(key_value) == 2:
                key, value = key_value
                key = key.strip().lower()
                value = value.strip()
                if '{' in value:  # Handle lists
                    value = value.replace('{', '').replace('}', '').split(',')
                    value = [v.strip() for v in value]
                metadata[key] = value
    return metadata

def map_envi_data_type(envi_data_type):
    """Map ENVI data types to numpy data types."""
    data_type_mapping = {
        '1': np.uint8,
        '2': np.int16,
        '3': np.int32,
        '4': np.float32,  # Your data type
        '5': np.float64,
        '12': np.uint16,
        '13': np.uint32,
        '14': np.int64,
        '15': np.uint64
    }
    return data_type_mapping.get(envi_data_type, None)

def load_binary_file(binary_file, hdr_metadata):
    """Load binary data using the metadata from the .hdr file."""
    nrows = int(hdr_metadata['lines'])
    ncols = int(hdr_metadata['samples'])
    nbands = int(hdr_metadata['bands'])
    dtype = map_envi_data_type(hdr_metadata['data type'][0])  # Map the ENVI data type

    if dtype is None:
        raise ValueError(f"Unsupported or unknown data type: {hdr_metadata['data type'][0]}")

    interleave = hdr_metadata['interleave'].lower()

    # Read the binary data
    data = np.fromfile(binary_file, dtype=dtype)

    # Reshape the data based on interleave format
    if interleave == 'bil':
        data = data.reshape((nrows, nbands, ncols)).transpose(1, 0, 2)  # Band Interleaved by Line
    elif interleave == 'bsq':
        data = data.reshape((nbands, nrows, ncols))  # Band Sequential
    elif interleave == 'bip':
        data = data.reshape((nrows, ncols, nbands)).transpose(2, 0, 1)  # Band Interleaved by Pixel
    else:
        raise ValueError(f"Unsupported interleave format: {interleave}")

    return data

def convert_to_netcdf_cdf4(binary_file, hdr_file, output_nc_file):
    """Convert binary and .hdr file data to a compressed NetCDF4-CDF4 file."""
    # Parse metadata from the .hdr file
    hdr_metadata = parse_hdr_file(hdr_file)

    # Load binary data
    data = load_binary_file(binary_file, hdr_metadata)

    # Convert to xarray Dataset
    ds = xr.Dataset(
        {"data": (["band", "y", "x"], data)},
        coords={
            "band": np.arange(1, data.shape[0] + 1),
            "y": np.arange(data.shape[1]),
            "x": np.arange(data.shape[2])
        },
        attrs={"description": "Binary data converted to NetCDF4-CDF4"}
    )

    # Set compression settings for NetCDF4-CDF4
    compression = {
        'data': {
            'zlib': True,       # Use zlib compression
            'complevel': 5,     # Set compression level (1-9)
            'shuffle': True     # Enable shuffle filter to improve compression efficiency
        }
    }

    # Save the dataset to NetCDF4-CDF4
    ds.to_netcdf(output_nc_file, format='NETCDF4_CLASSIC', encoding=compression)
    print(f"Saved compressed NetCDF4-CDF4 file to: {output_nc_file}")

# Example usage
binary_file = '/Users/kitlewers/Desktop/naive_compression/imagery/ang20231109t092617_027_L2A_OE_main_27577724_RFL_ORT'  # Replace with your binary file path
hdr_file = '/Users/kitlewers/Desktop/naive_compression/imagery/ang20231109t092617_027_L2A_OE_main_27577724_RFL_ORT.hdr'  # Replace with your header file path
output_nc_file = '/Users/kitlewers/Desktop/naive_compression/imagery/output_data.nc'

convert_to_netcdf_cdf4(binary_file, hdr_file, output_nc_file)
