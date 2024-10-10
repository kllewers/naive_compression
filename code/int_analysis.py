import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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
        '4': np.float32,
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

def count_decimal_places(data):
    """Count the number of decimal places for each value."""
    decimal_places = []
    for val in data.flatten():
        if val != -9999:  # Ignore invalid values
            if '.' in str(val):  # Only consider values with decimal points
                decimal_part = str(val).split('.')[-1]
                decimal_places.append(len(decimal_part))
            else:
                decimal_places.append(0)  # No decimal places for integers
    return np.array(decimal_places)

# Function to generate distribution plot and summary table
def analyze_decimal_places(binary_file, hdr_file):
    # Load binary data
    hdr_metadata = parse_hdr_file(hdr_file)
    data = load_binary_file(binary_file, hdr_metadata)

    # Count decimal places excluding -9999 values
    decimal_places = count_decimal_places(data)

    # Plot the distribution of decimal places
    plt.hist(decimal_places, bins=np.arange(0, max(decimal_places) + 2) - 0.5, edgecolor='black')
    plt.xlabel('Number of Decimal Places')
    plt.ylabel('Frequency')
    plt.title('Distribution of Decimal Places in the Data')
    plt.show()

    # Create a summary table of statistics
    summary_stats = {
        "Mean Decimal Places": [np.mean(decimal_places)],
        "Median Decimal Places": [np.median(decimal_places)],
        "Max Decimal Places": [np.max(decimal_places)],
        "Min Decimal Places": [np.min(decimal_places)],
        "Total Valid Values": [len(decimal_places)]
    }

    summary_df = pd.DataFrame(summary_stats)
    
    # Display the summary table
    import ace_tools as tools; tools.display_dataframe_to_user(name="Decimal Places Summary", dataframe=summary_df)

# Example usage
if __name__ == "__main__":
    binary_file = '/Users/kitlewers/Desktop/naive_compression/imagery/ang20231109t092617_027_L2A_OE_main_27577724_RFL_ORT'  # Replace with your binary file path
    hdr_file = '/Users/kitlewers/Desktop/naive_compression/imagery/ang20231109t092617_027_L2A_OE_main_27577724_RFL_ORT.hdr'  # Replace with your header file path
    
    analyze_decimal_places(binary_file, hdr_file)
