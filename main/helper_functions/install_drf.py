import os
import sys
import platform
import subprocess
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
import rpy2.robjects as robjects

# --- 1. Detect OS Codename (jammy, focal, buster, etc.) ---
try:
    # Read /etc/os-release to find the codename (e.g., "jammy" or "focal")
    with open("/etc/os-release") as f:
        lines = f.readlines()
        codename = None
        for line in lines:
            if line.startswith("VERSION_CODENAME="):
                codename = line.split("=")[1].strip().strip('"')
                break
            
        # Fallback if VERSION_CODENAME isn't there
        if not codename:
            for line in lines:
                if line.startswith("PRETTY_NAME="):
                    if "Ubuntu 22.04" in line: codename = "jammy"
                    elif "Ubuntu 20.04" in line: codename = "focal"
                    elif "Debian" in line and "11" in line: codename = "bullseye"
                    elif "Debian" in line and "10" in line: codename = "buster"
                    break
                    
    if not codename:
        print(">> Warning: Could not determine OS. Defaulting to 'jammy' (Ubuntu 22.04).")
        codename = "jammy"
except Exception:
    codename = "jammy"

print(f">> Detected Container OS: {codename}")

# --- 2. Setup Local Library ---
r_libs_user = os.path.expanduser("~/R_libs")
os.makedirs(r_libs_user, exist_ok=True)
robjects.r(f'.libPaths(c("{r_libs_user}", .libPaths()))')

# --- 3. Configure Binary Repository ---
# Posit Package Manager URL for this specific OS
binary_repo = f"https://packagemanager.posit.co/cran/__linux__/{codename}/latest"

# CRITICAL: Set the User-Agent so the server delivers binaries, not source
# R requires this specific string format to accept Linux binaries
agent_cmd = f'options(HTTPUserAgent = sprintf("R/%s R (%s)", getRversion(), paste(getRversion(), R.version$platform, arch = R.version$arch, os = R.version$os)))'
robjects.r(agent_cmd)

# Also force the repo option
robjects.r(f'options(repos = c(CRAN = "{binary_repo}"))')

print(f">> Installing from Binary Repo: {binary_repo}")
print(f">> Target Directory: {r_libs_user}")

# --- 4. Install ---
utils = rpackages.importr('utils')

try:
    # Attempt installation
    utils.install_packages(StrVector(['drf']), lib=r_libs_user)
    
    # --- 5. Verify ---
    if rpackages.isinstalled("drf", lib_loc=r_libs_user):
        print("\n>>> SUCCESS: 'drf' is installed!")
        # Quick check if it loads
        try:
            robjects.r(f"library(drf, lib.loc='{r_libs_user}')")
            print(">>> Verification: Library loads successfully.")
        except Exception as e:
            print(f">>> Warning: Installed but failed to load: {e}")
    else:
        print("\n>>> FAILURE: Installation finished but package not found.")
        sys.exit(1)

except Exception as e:
    print(f"\n>>> CRITICAL ERROR: {e}")
    sys.exit(1)