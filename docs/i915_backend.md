# I915 Backend for Tinygrad

This document explains how to use the Intel i915 backend in Tinygrad, which provides direct communication with Intel GPUs through the i915 driver using IOCTLs.

## Overview

The i915 backend allows Tinygrad to interact directly with Intel GPUs through the Linux i915 driver interface instead of using OpenCL. This approach offers several advantages:

1. **Direct Hardware Control**: Lower-level access to the GPU resources
2. **Parallel Operations**: Better support for operations like beam search
3. **Performance Optimization**: Potential for better performance by avoiding OpenCL overhead
4. **Customization**: Ability to implement specialized operations tailored to Intel GPUs

## Requirements

- Linux operating system
- Intel GPU with a compatible i915 driver
- Appropriate permissions to access `/dev/dri/renderD*` devices

## Usage

### Enabling the Backend

To use the i915 backend, set the `I915` environment variable to 1:

```bash
export I915=1
```

Or prefix your command:

```bash
I915=1 python your_script.py
```

### Basic Example

Here's a simple example of using the i915 backend:

```python
from tinygrad import Tensor, dtypes

# Create tensors on the i915 device
a = Tensor([1, 2, 3, 4], device="I915")
b = Tensor([5, 6, 7, 8], device="I915")

# Perform operations
c = a + b

# Convert back to numpy
result = c.numpy()
print(result)  # [6, 8, 10, 12]
```

### Selecting a Specific GPU

If you have multiple Intel GPUs, you can select a specific one using the format "I915:N" where N is the device number:

```python
# Use the second Intel GPU
tensor = Tensor([1, 2, 3], device="I915:1")
```

The device number corresponds to the render nodes in `/dev/dri/`, where `/dev/dri/renderD128` is device 0, `/dev/dri/renderD129` is device 1, etc.

## Advanced Usage: Beam Search

The i915 backend is particularly useful for parallel operations like beam search in language models. See the example in `examples/i915_beam_search.py` for a demonstration:

```bash
I915=1 python examples/i915_beam_search.py --beam_width 8 --max_length 30
```

This example showcases how to perform parallel beam search efficiently using the i915 backend.

## Implementation Details

The backend consists of several key components:

1. **I915Device**: Main class that interfaces with the Intel GPU
2. **I915Allocator**: Handles memory allocation and transfers
3. **I915Program**: Represents a compiled kernel program
4. **I915Compiler**: Compiles kernel programs for the Intel GPU

The implementation uses IOCTLs to communicate directly with the i915 driver, offering lower-level access than what's available through OpenCL.

## Debugging

To enable debug output, set the `DEBUG` environment variable:

```bash
DEBUG=1 I915=1 python your_script.py
```

Higher values provide more detailed information:

```bash
DEBUG=2 I915=1 python your_script.py  # More verbose output
```

## Limitations

Current limitations of the i915 backend include:

1. Early development stage - some features may be incomplete
2. Limited to Linux systems with compatible i915 drivers
3. May require GPU-specific optimizations for best performance

## Future Work

Future improvements for the i915 backend include:

1. Full support for all Tinygrad operations 
2. Optimized kernels for common operations
3. Better handling of different Intel GPU generations
4. Improved error reporting and diagnostics
5. Expanded documentation and examples

## Getting Involved

Contributions to improve the i915 backend are welcome! Areas where help is needed include:

1. Testing on different Intel GPU models
2. Performance optimization
3. Adding support for more operations
4. Improving documentation and examples
5. Integration with Intel's oneAPI where appropriate

For development discussions, please see the main Tinygrad channels.