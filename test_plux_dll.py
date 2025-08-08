import os
import sys
import platform
import ctypes

# Determine the correct directory
python_version = ''.join(platform.python_version().split('.')[:2])
plux_dir = f"PLUX-API-Python3/Win64_{python_version}"

# Fallback to available versions
if not os.path.exists(plux_dir):
    fallback_versions = ["Win64_312", "Win64_310", "Win64_39", "Win64_38", "Win64_37"]
    for version in fallback_versions:
        if os.path.exists(f"PLUX-API-Python3/{version}"):
            plux_dir = f"PLUX-API-Python3/{version}"
            print(f"Using {version}")
            break

# Try to load the DLLs manually first
dll_path = os.path.abspath(plux_dir)
print(f"DLL path: {dll_path}")

try:
    # Load dependencies first
    msvcr100 = ctypes.cdll.LoadLibrary(os.path.join(dll_path, "msvcr100.dll"))
    print("OK Loaded msvcr100.dll")
    
    msvcp100 = ctypes.cdll.LoadLibrary(os.path.join(dll_path, "msvcp100.dll"))
    print("OK Loaded msvcp100.dll")
    
    ft4222 = ctypes.cdll.LoadLibrary(os.path.join(dll_path, "LibFT4222-64.dll"))
    print("OK Loaded LibFT4222-64.dll")
    
    ft4222ab = ctypes.cdll.LoadLibrary(os.path.join(dll_path, "LibFT4222AB-64.dll"))
    print("OK Loaded LibFT4222AB-64.dll")
    
    print("All DLLs loaded successfully!")
    
except Exception as e:
    print(f"Error loading DLLs: {e}")

# Now try to import plux
sys.path.append(plux_dir)
os.environ["PATH"] = dll_path + ";" + os.environ["PATH"]

try:
    import plux
    print("OK Successfully imported plux!")
    
    # Test basic functionality
    print("plux module attributes:")
    print(dir(plux))
    
except Exception as e:
    print(f"Error importing plux: {e}")
    
    # Try different approaches
    print("\nTrying alternative approaches...")
    
    # Method 1: Change working directory
    original_cwd = os.getcwd()
    try:
        os.chdir(dll_path)
        import plux
        print("OK Success with directory change method!")
    except Exception as e2:
        print(f"Directory change method failed: {e2}")
    finally:
        os.chdir(original_cwd)
    
    # Method 2: Use AddDllDirectory (Windows 10+)
    try:
        import ctypes.wintypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetDllDirectoryW(dll_path)
        import plux
        print("OK Success with SetDllDirectory method!")
    except Exception as e3:
        print(f"SetDllDirectory method failed: {e3}")