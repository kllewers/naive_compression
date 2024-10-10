import numpy as np
import xarray as xr
import xbitinfo as xb
import os
import julia
julia.install()
os.environ["PATH"] += os.pathsep + "/Users/kitlewers/.juliaup/bin"


# Step 1: Parse the ENVI .hdr file (from your existing code)
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

# Step 2: Map ENVI data types to numpy types (from your existing code)
def map_envi_data_type(envi_data_type):
    """Map ENVI data types to numpy data types."""
    data_type_mapping = {
        '1': np.uint8,
        '2': np.int16,
        '3': np.int32,
        '4': np.float32,
        '5': np.float64,
        '12': np.uint16,
        '13': np.uint32,
        '14': np.int64,
        '15': np.uint64
    }
    return data_type_mapping.get(envi_data_type, None)

# Step 3: Load the binary file (from your existing code)
def load_binary_file(binary_file, hdr_metadata):
    """Load binary data using the metadata from the .hdr file."""
    nrows = int(hdr_metadata['lines'])
    ncols = int(hdr_metadata['samples'])
    nbands = int(hdr_metadata['bands'])
    dtype = map_envi_data_type(hdr_metadata['data type'][0])

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

# Step 4: Convert the binary data to an xarray Dataset
def convert_to_xarray(binary_data, hdr_metadata):
    nrows = int(hdr_metadata['lines'])
    ncols = int(hdr_metadata['samples'])
    nbands = int(hdr_metadata['bands'])
    
    # Create an xarray Dataset
    ds = xr.Dataset(
        {"data": (["band", "y", "x"], binary_data)},
        coords={
            "band": np.arange(1, nbands + 1),
            "y": np.arange(nrows),
            "x": np.arange(ncols)
        }
    )
    return ds

# Step 5: Use xbitinfo for compression with chunking
def compress_with_xbitinfo(ds, output_nc_file, inflevel=0.99, chunksizes=(1, 100, 100)):
    # Analyze bit information
    bitinfo = xb.get_bitinformation(ds, dim="band")
    
    # Get the number of bits to keep for the specified information level
    keepbits = xb.get_keepbits(bitinfo, inflevel=inflevel)
    
    # Apply bit rounding to the dataset
    ds_bitrounded = xb.xr_bitround(ds, keepbits)
    
    # Set chunk sizes and compression settings
    compression = {
        'data': {
            'zlib': True,
            'complevel': 5,
            'shuffle': True,
            'chunksizes': chunksizes
        }
    }
    
    # Save the bit-rounded dataset with compression and chunking
    ds_bitrounded.to_netcdf(output_nc_file, format='NETCDF4', encoding=compression)

# Example usage
if __name__ == "__main__":
    binary_file = '/Users/kitlewers/Desktop/naive_compression/imagery/ang20231109t092617_027_L2A_OE_main_27577724_RFL_ORT'
    hdr_file = '/Users/kitlewers/Desktop/naive_compression/imagery/ang20231109t092617_027_L2A_OE_main_27577724_RFL_ORT.hdr'
    output_nc_file = '/Users/kitlewers/Desktop/naive_compression/imagery/lossy_xbitinfo_compression.nc'

    # Step 1: Parse the .hdr file to extract metadata
    hdr_metadata = parse_hdr_file(hdr_file)
    
    # Step 2: Load the binary data using the metadata
    binary_data = load_binary_file(binary_file, hdr_metadata)
    
    # Step 3: Convert the loaded binary data to an xarray Dataset
    ds = convert_to_xarray(binary_data, hdr_metadata)
    
    # Step 4: Use xbitinfo to analyze, compress, and ensure consistent chunking
    compress_with_xbitinfo(ds, output_nc_file, inflevel=0.99, chunksizes=(1, 100, 100))
