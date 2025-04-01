from __future__ import annotations
import os, ctypes, functools, hashlib, time
import numpy as np
from typing import Optional, cast, Any
from tinygrad.helpers import DEBUG, getenv, OSX, diskcache_get, diskcache_put
from tinygrad.dtype import DType, dtypes
from tinygrad.device import Compiled, LRUAllocator, CompileError, Compiler, Buffer, BufferSpec
from tinygrad.renderer.cstyle import IntelRenderer
from tinygrad.runtime.autogen import i915
from tinygrad.helpers import init_c_var, cpu_time_execution, from_mv, mv_address

# Custom exceptions for I915 backend
class I915Error(Exception): pass
class I915CompileError(CompileError): pass

# Helper function for error checking
def check_error(ret, err_msg="I915 Error"):
    if ret != 0:
        raise I915Error(f"{err_msg}: {os.strerror(ctypes.get_errno())}")
    return ret

# Program class for I915 backend
class I915Program:
    def __init__(self, device:I915Device, name:str, lib:bytes):
        self.device, self.name, self.lib = device, name, lib
        kernel_size = len(lib)
        
        if DEBUG >= 1: print(f"I915Program: creating program '{name}' with {kernel_size} bytes")
        
        # Create a buffer for the kernel code
        self.kernel_handle = i915.gem_create(self.device.fd, kernel_size)
        if DEBUG >= 2: print(f"I915Program: created kernel buffer with handle {self.kernel_handle}")
        
        # Map the kernel buffer to memory and copy the kernel code
        self.kernel_addr = i915.gem_mmap(self.device.fd, self.kernel_handle, kernel_size)
        if DEBUG >= 2: print(f"I915Program: mapped kernel buffer to address {self.kernel_addr}")
        
        ctypes.memmove(self.kernel_addr, lib, kernel_size)
        
        # Create an exec_object structure for the kernel
        self.exec_object = i915.drm_i915_gem_exec_object2(
            handle=self.kernel_handle,
            relocation_count=0,
            relocs_ptr=0,
            alignment=0,
            offset=0,
            flags=0
        )
    
    def __del__(self):
        try:
            if hasattr(self, 'kernel_handle'):
                if DEBUG >= 2: print(f"I915Program: closing kernel handle {self.kernel_handle}")
                i915.gem_close(self.device.fd, self.kernel_handle)
        except Exception as e:
            if DEBUG >= 1: print(f"I915Program: error cleaning up kernel: {e}")
    
    def __call__(self, *bufs:tuple[int, BufferSpec], global_size:tuple[int,int,int]=(1,1,1), 
                 local_size:Optional[tuple[int,int,int]]=None, vals:tuple[int, ...]=(), wait=False) -> Optional[float]:
        if DEBUG >= 2: 
            print(f"I915Program: executing kernel '{self.name}' with {len(bufs)} buffers")
            print(f"I915Program: global_size={global_size}, local_size={local_size}, vals={vals}")
        
        # Prepare buffer exec objects
        exec_objects = (i915.drm_i915_gem_exec_object2 * (len(bufs) + 1))()
        
        # First object is the kernel
        exec_objects[0] = self.exec_object
        
        # Add buffer objects
        for i, (buf_handle, _) in enumerate(bufs):
            exec_objects[i+1] = i915.drm_i915_gem_exec_object2(
                handle=buf_handle,
                relocation_count=0,
                relocs_ptr=0,
                alignment=0,
                offset=0,
                flags=0
            )
        
        # Execute the kernel
        start_time = None
        if wait:
            start_time = cpu_time_execution.time_ns()
            
        i915.gem_execbuffer2(
            self.device.fd,
            exec_objects,
            len(exec_objects),
            0,  # batch_start_offset
            len(self.lib),  # batch_len
            flags=i915.ExecFlags.I915_EXEC_COMPUTE,
            ctx_id=self.device.ctx_id
        )
        
        # Wait for completion if requested
        if wait:
            i915.gem_wait(self.device.fd, self.kernel_handle)
            end_time = cpu_time_execution.time_ns()
            return (end_time - start_time) * 1e-9
        
        return None

class I915Allocator(LRUAllocator):
    def __init__(self, dev:I915Device):
        self.dev = dev
        if DEBUG >= 1: print(f"I915Allocator: initializing for device {dev.device_name}")
        self.buffer_cache = {}  # Cache for buffer data - only used as a fallback
        super().__init__()
        
    def _alloc(self, size:int, options:BufferSpec) -> tuple[int, BufferSpec]:
        if DEBUG >= 2: print(f"I915Allocator: allocating buffer of size {size} bytes")
        
        # Create a GEM buffer
        handle = i915.gem_create(self.dev.fd, size)
        if DEBUG >= 2: print(f"I915Allocator: allocated buffer with handle {handle}")
        
        return (handle, options)
        
    def _free(self, opaque:tuple[int, BufferSpec], options:BufferSpec):
        handle, _ = opaque
        if DEBUG >= 2: print(f"I915Allocator: freeing buffer with handle {handle}")
        
        # Remove from the buffer cache if it's there
        if handle in self.buffer_cache:
            del self.buffer_cache[handle]
            
        # Close the GEM buffer
        i915.gem_close(self.dev.fd, handle)
        
    def _copyin(self, dest:tuple[int, BufferSpec], src:memoryview):
        handle, _ = dest
        size = len(src) * src.itemsize
        
        if DEBUG >= 2: print(f"I915Allocator: copying in {size} bytes to buffer with handle {handle}")
        
        # Store a copy in our cache as a fallback
        src_bytes = bytes(src)
        self.buffer_cache[handle] = src_bytes
        
        # Map the buffer and copy the data
        addr = i915.gem_mmap(self.dev.fd, handle, size)
        
        # Copy data to the mapped buffer
        ctypes.memmove(addr, from_mv(src), size)
        
    def _copyout(self, dest:memoryview, src:tuple[int, BufferSpec]):
        handle, _ = src
        size = len(dest) * dest.itemsize
        
        if DEBUG >= 2: print(f"I915Allocator: copying out {size} bytes from buffer with handle {handle}")
        
        # Map the buffer and copy the data from it
        addr = i915.gem_mmap(self.dev.fd, handle, size)
        
        # Copy data from the mapped buffer to the destination
        ctypes.memmove(from_mv(dest), addr, size)
        
        # Synchronize to ensure the operation has completed
        self.dev.synchronize()

class I915Compiler(Compiler):
    def __init__(self, dev:I915Device, compile_key:str):
        self.dev = dev
        super().__init__(f"compile_i915_{compile_key}")
        
    def compile(self, src:str) -> bytes:
        # This is a placeholder - an actual implementation would need to:
        # 1. Compile OpenCL C to Intel GPU ISA, or
        # 2. Directly generate Intel GPU ISA from high-level IR
        
        # For now, we'll just return the source as bytes
        # In a real implementation, this would use Intel's compiler tools
        try:
            # Placeholder for actual compilation
            return src.encode()
        except Exception as e:
            raise I915CompileError(f"I915 Compile Error: {e}")
        
    def disassemble(self, lib:bytes):
        # Placeholder for disassembly
        if DEBUG >= 2:
            print("I915 kernel disassembly not implemented")

class I915Device(Compiled):
    def __init__(self, device:str=""):
        # Open the i915 device
        node_num = 0  # Default to first GPU
        if ":" in device:
            node_num = int(device.split(":")[1])
            
        device_path = f"/dev/dri/renderD{node_num + 128}"
        if DEBUG >= 1: print(f"I915Device: attempting to open {device_path}")
        
        # Check if we have access to the device
        if not os.path.exists(device_path):
            raise OSError(f"I915 device {device_path} does not exist")
            
        if not os.access(device_path, os.R_OK | os.W_OK):
            import getpass
            username = getpass.getuser()
            raise OSError(f"No permission to access I915 device {device_path}. "
                          f"Add your user to the 'render' group with: sudo usermod -a -G render {username} "
                          f"and then log out and log back in.")
        
        try:
            self.fd = os.open(device_path, os.O_RDWR)
            if self.fd < 0:
                raise OSError(f"Failed to open i915 device: {os.strerror(ctypes.get_errno())}")
                
            if DEBUG >= 1: print(f"I915Device: successfully opened fd={self.fd}")
            
            # Verify we can perform basic operations with this device
            try:
                # Try a simple GEM buffer creation to verify the device works
                test_handle = i915.gem_create(self.fd, 4096)  # 4KB test buffer
                if DEBUG >= 1: print(f"I915Device: verified device with test buffer handle={test_handle}")
                
                # Clean up test buffer
                i915.gem_close(self.fd, test_handle)
            except OSError as e:
                # Common error: We can open the device but not create buffers (permission denied)
                if "Permission denied" in str(e):
                    import getpass
                    username = getpass.getuser()
                    raise OSError(f"Permission denied when creating i915 buffer. "
                                f"Add your user to the 'render' group with: sudo usermod -a -G render {username} "
                                f"and then log out and log back in.")
                else:
                    raise OSError(f"I915 device verification failed. This might not be an Intel GPU with i915 driver: {e}")
            
            # Create a GPU context
            try:
                self.ctx_id = i915.gem_context_create(self.fd)
                if DEBUG >= 1: print(f"I915Device: created context with id={self.ctx_id}")
            except Exception as e:
                if DEBUG >= 1: print(f"I915Device: context creation failed with {e}, using default context 0")
                self.ctx_id = 0  # Use default context if creation fails
            
            # Get device info
            self.device_name = f"Intel GPU #{node_num}"
            if DEBUG >= 1: print(f"I915Device: opened {self.device_name}")
        except Exception as e:
            if DEBUG >= 1: print(f"I915Device: initialization failed: {e}")
            raise
        
        # Create a unique key for the compiler cache
        compile_key = hashlib.md5(f"{self.device_name}".encode()).hexdigest()
        
        # Create the renderer, allocator and compiler
        renderer = IntelRenderer()
        allocator = I915Allocator(self)
        compiler = I915Compiler(self, compile_key)
        
        super().__init__(device, allocator, renderer, compiler, functools.partial(I915Program, self))

    def synchronize(self):
        # Synchronize all pending operations
        # This is a simplified implementation
        pass
        
    def _at_profile_finalize(self):
        # Called at profile finalization
        pass
        
    def finalize(self):
        # Clean up resources
        if hasattr(self, 'ctx_id'):
            i915.gem_context_destroy(self.fd, self.ctx_id)
            
        if hasattr(self, 'fd') and self.fd >= 0:
            os.close(self.fd)