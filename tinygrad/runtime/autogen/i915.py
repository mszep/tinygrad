from __future__ import annotations
import ctypes, ctypes.util, os
from enum import IntEnum

# Load libc for ioctl support
libc = ctypes.CDLL(ctypes.util.find_library("c"))

# Define ioctl functions
_IOC_NRBITS = 8
_IOC_TYPEBITS = 8
_IOC_SIZEBITS = 14
_IOC_DIRBITS = 2

_IOC_NRSHIFT = 0
_IOC_TYPESHIFT = _IOC_NRSHIFT + _IOC_NRBITS
_IOC_SIZESHIFT = _IOC_TYPESHIFT + _IOC_TYPEBITS
_IOC_DIRSHIFT = _IOC_SIZESHIFT + _IOC_SIZEBITS

_IOC_NONE = 0
_IOC_WRITE = 1
_IOC_READ = 2

def _IOC(dir, type, nr, size):
    return (dir << _IOC_DIRSHIFT) | (type << _IOC_TYPESHIFT) | (nr << _IOC_NRSHIFT) | (size << _IOC_SIZESHIFT)

def _IO(type, nr): return _IOC(_IOC_NONE, type, nr, 0)
def _IOR(type, nr, size): return _IOC(_IOC_READ, type, nr, ctypes.sizeof(size))
def _IOW(type, nr, size): return _IOC(_IOC_WRITE, type, nr, ctypes.sizeof(size))
def _IOWR(type, nr, size): return _IOC(_IOC_READ | _IOC_WRITE, type, nr, ctypes.sizeof(size))

# Define I915 IOCTL commands
I915_IOCTL_BASE = ord('d')
I915_IOCTL_GEM_CREATE = _IOWR(I915_IOCTL_BASE, 0x2, drm_i915_gem_create)
I915_IOCTL_GEM_CLOSE = _IOW(I915_IOCTL_BASE, 0x3, drm_i915_gem_close)
I915_IOCTL_GEM_MMAP = _IOWR(I915_IOCTL_BASE, 0x7, drm_i915_gem_mmap)
I915_IOCTL_GEM_EXECBUFFER2 = _IOWR(I915_IOCTL_BASE, 0x10, drm_i915_gem_execbuffer2)
I915_IOCTL_GEM_WAIT = _IOW(I915_IOCTL_BASE, 0x16, drm_i915_gem_wait)
I915_IOCTL_GEM_CONTEXT_CREATE = _IOWR(I915_IOCTL_BASE, 0x1d, drm_i915_gem_context_create)
I915_IOCTL_GEM_CONTEXT_DESTROY = _IOW(I915_IOCTL_BASE, 0x1e, drm_i915_gem_context_destroy)
I915_IOCTL_GEM_MADVISE = _IOW(I915_IOCTL_BASE, 0x11, ctypes.c_void_p)
I915_IOCTL_REG_READ = _IOWR(I915_IOCTL_BASE, 0x31, ctypes.c_void_p)
I915_IOCTL_GET_PARAM = _IOWR(I915_IOCTL_BASE, 0x6, ctypes.c_void_p)
I915_IOCTL_GEM_USERPTR = _IOWR(I915_IOCTL_BASE, 0x33, drm_i915_gem_userptr)

# Define I915 GEM Execbuffer flags
class ExecFlags(IntEnum):
    I915_EXEC_DEFAULT = 0
    I915_EXEC_RENDER = 1
    I915_EXEC_BLT = 2
    I915_EXEC_BSD = 3
    I915_EXEC_VEBOX = 4
    I915_EXEC_COMPUTE = 5
    
    I915_EXEC_FENCE_IN = 0x1
    I915_EXEC_FENCE_OUT = 0x2
    I915_EXEC_BATCH_FIRST = 0x4
    I915_EXEC_FENCE_ARRAY = 0x8
    I915_EXEC_BSD_MASK = 0x3
    I915_EXEC_BSD_RING1 = 0x1
    I915_EXEC_BSD_RING2 = 0x2
    I915_EXEC_NO_RELOC = 0x10
    I915_EXEC_HANDLE_LUT = 0x20
    I915_EXEC_SECURE = 0x80
    I915_EXEC_PARALLEL_SUBMIT = 0x100

# Structures for i915 IOCTLs
class drm_i915_gem_create(ctypes.Structure):
    _fields_ = [
        ("size", ctypes.c_uint64),
        ("handle", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
    ]

class drm_i915_gem_close(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
    ]

class drm_i915_gem_mmap(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
        ("offset", ctypes.c_uint64),
        ("size", ctypes.c_uint64),
        ("addr_ptr", ctypes.c_uint64),
        ("flags", ctypes.c_uint64),
    ]

class drm_i915_gem_context_create(ctypes.Structure):
    _fields_ = [
        ("ctx_id", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
    ]

class drm_i915_gem_context_destroy(ctypes.Structure):
    _fields_ = [
        ("ctx_id", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
    ]

class drm_i915_gem_userptr(ctypes.Structure):
    _fields_ = [
        ("user_ptr", ctypes.c_uint64),
        ("user_size", ctypes.c_uint64),
        ("flags", ctypes.c_uint32),
        ("handle", ctypes.c_uint32),
    ]

class drm_i915_gem_exec_object2(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("relocation_count", ctypes.c_uint32),
        ("relocs_ptr", ctypes.c_uint64),
        ("alignment", ctypes.c_uint64),
        ("offset", ctypes.c_uint64),
        ("flags", ctypes.c_uint64),
        ("rsvd1", ctypes.c_uint64),
        ("rsvd2", ctypes.c_uint64),
    ]

class drm_i915_gem_execbuffer2(ctypes.Structure):
    _fields_ = [
        ("buffers_ptr", ctypes.c_uint64),
        ("buffer_count", ctypes.c_uint32),
        ("batch_start_offset", ctypes.c_uint32),
        ("batch_len", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("rsvd1", ctypes.c_uint64),
        ("rsvd2", ctypes.c_uint64),
        ("ctx_id", ctypes.c_uint32),
        ("rsvd3", ctypes.c_uint32),
        ("dependencies_ptr", ctypes.c_uint64),
        ("dependency_count", ctypes.c_uint32),
        ("pad", ctypes.c_uint32),
    ]

class drm_i915_gem_wait(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("timeout_ns", ctypes.c_int64),
    ]

# Helper functions for I915 IOCTLs
def gem_create(fd, size):
    create = drm_i915_gem_create(size=size, handle=0)
    result = libc.ioctl(fd, I915_IOCTL_GEM_CREATE, ctypes.byref(create))
    if result != 0:
        errno = ctypes.get_errno()
        if errno != 0:  # Only raise if there's an actual error code
            raise OSError(f"I915 GEM CREATE failed: {os.strerror(errno)}")
    
    # Check if the resulting handle is valid (non-zero)
    if create.handle == 0:
        raise OSError(f"I915 GEM CREATE returned zero handle")
    
    return create.handle

def gem_close(fd, handle):
    close = drm_i915_gem_close(handle=handle)
    result = libc.ioctl(fd, I915_IOCTL_GEM_CLOSE, ctypes.byref(close))
    if result != 0:
        raise OSError(f"I915 GEM CLOSE failed: {os.strerror(ctypes.get_errno())}")

def gem_mmap(fd, handle, size, offset=0, flags=0):
    mmap_arg = drm_i915_gem_mmap(handle=handle, offset=offset, size=size, addr_ptr=0, flags=flags)
    result = libc.ioctl(fd, I915_IOCTL_GEM_MMAP, ctypes.byref(mmap_arg))
    if result != 0:
        errno = ctypes.get_errno()
        if errno != 0:  # Only raise if there's an actual error code
            raise OSError(f"I915 GEM MMAP failed: {os.strerror(errno)}")
    
    # Check if the resulting address is valid (non-zero)
    if mmap_arg.addr_ptr == 0:
        raise OSError(f"I915 GEM MMAP returned zero address for handle {handle}")
        
    return mmap_arg.addr_ptr

def gem_context_create(fd):
    ctx_create = drm_i915_gem_context_create(ctx_id=0)
    result = libc.ioctl(fd, I915_IOCTL_GEM_CONTEXT_CREATE, ctypes.byref(ctx_create))
    if result != 0:
        errno = ctypes.get_errno()
        if errno != 0:  # Only raise if there's an actual error code
            raise OSError(f"I915 GEM CONTEXT CREATE failed: {os.strerror(errno)}")
    return ctx_create.ctx_id

def gem_context_destroy(fd, ctx_id):
    ctx_destroy = drm_i915_gem_context_destroy(ctx_id=ctx_id)
    result = libc.ioctl(fd, I915_IOCTL_GEM_CONTEXT_DESTROY, ctypes.byref(ctx_destroy))
    if result != 0:
        errno = ctypes.get_errno()
        if errno != 0:  # Only raise if there's an actual error code
            raise OSError(f"I915 GEM CONTEXT DESTROY failed: {os.strerror(errno)}")

def gem_userptr(fd, user_ptr, user_size, flags=0):
    userptr_arg = drm_i915_gem_userptr(user_ptr=user_ptr, user_size=user_size, flags=flags, handle=0)
    result = libc.ioctl(fd, I915_IOCTL_GEM_USERPTR, ctypes.byref(userptr_arg))
    if result != 0:
        raise OSError(f"I915 GEM USERPTR failed: {os.strerror(ctypes.get_errno())}")
    return userptr_arg.handle

def gem_execbuffer2(fd, objects, buffer_count, batch_start_offset, batch_len, flags=0, ctx_id=0):
    exec_buffer = drm_i915_gem_execbuffer2(
        buffers_ptr=ctypes.addressof(objects),
        buffer_count=buffer_count,
        batch_start_offset=batch_start_offset,
        batch_len=batch_len,
        flags=flags,
        ctx_id=ctx_id
    )
    result = libc.ioctl(fd, I915_IOCTL_GEM_EXECBUFFER2, ctypes.byref(exec_buffer))
    if result != 0:
        raise OSError(f"I915 GEM EXECBUFFER2 failed: {os.strerror(ctypes.get_errno())}")
    return result

def gem_wait(fd, handle, timeout_ns=1000000000):
    wait_arg = drm_i915_gem_wait(handle=handle, flags=0, timeout_ns=timeout_ns)
    result = libc.ioctl(fd, I915_IOCTL_GEM_WAIT, ctypes.byref(wait_arg))
    if result != 0:
        raise OSError(f"I915 GEM WAIT failed: {os.strerror(ctypes.get_errno())}")
    return result