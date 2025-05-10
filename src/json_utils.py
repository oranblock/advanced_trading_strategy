import json
import numpy as np

def convert_numpy_types(obj):
    """
    Convert NumPy types to Python native types for JSON serialization.
    
    Args:
        obj: Object containing NumPy data types
        
    Returns:
        Object with NumPy types converted to Python native types
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return convert_numpy_types(obj.tolist())
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj

def save_json(obj, filepath, indent=4):
    """
    Save object to JSON file with NumPy type conversion.
    
    Args:
        obj: Object to serialize
        filepath: Path to save the JSON file
        indent: Indentation for pretty printing (default: 4)
    """
    # Convert NumPy types to Python native types
    obj_serializable = convert_numpy_types(obj)
    
    # Write to file
    with open(filepath, 'w') as f:
        json.dump(obj_serializable, f, indent=indent)