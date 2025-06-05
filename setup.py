import os
import sys
import subprocess
import platform
import time
from pathlib import Path

def create_venv():
    """Create and activate a virtual environment."""
    print("Creating virtual environment...")
    
    # Check if venv already exists
    venv_dir = Path("venv")
    if venv_dir.exists():
        print("Virtual environment already exists.")
    else:
        # Create virtual environment
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("Virtual environment created successfully.")
    
    # Get the path to the virtual environment Python executable
    if platform.system() == "Windows":
        venv_python = str(venv_dir / "Scripts" / "python.exe")
        venv_pip = str(venv_dir / "Scripts" / "pip.exe")
    else:
        venv_python = str(venv_dir / "bin" / "python")
        venv_pip = str(venv_dir / "bin" / "pip")
    
    # Upgrade pip and install setuptools first
    print("Upgrading pip and installing setuptools...")
    subprocess.run([venv_python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], check=True)
    
    # Install basic dependencies for downloading files
    print("Installing basic dependencies...")
    subprocess.run([venv_pip, "install", "requests", "tqdm", "gdown"], check=True)
    
    return venv_python, venv_pip

def download_file(url, output_path, max_retries=3, use_gdown=False):
    """
    Download a file using Python requests with retry mechanism.
    Args:
        url: URL of the file to download
        output_path: Path where to save the file
        max_retries: Maximum number of retry attempts
        use_gdown: Whether to use gdown for Google Drive links
    """
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # If using gdown for Google Drive links
    if use_gdown:
        try:
            # Import gdown in the context where it should be available
            import sys
            import subprocess
            
            # Get the virtual environment Python path
            venv_dir = Path("venv")
            if platform.system() == "Windows":
                venv_python = str(venv_dir / "Scripts" / "python.exe")
            else:
                venv_python = str(venv_dir / "bin" / "python")
            
            print(f"Downloading from Google Drive: {url}")
            # Use the virtual environment's Python to run gdown
            result = subprocess.run([
                venv_python, "-c", 
                f"import gdown; gdown.download('{url}', '{output_path}', quiet=False)"
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                print(f"Download completed: {output_path}")
                return True
            else:
                print(f"Download failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"Download with gdown failed: {e}")
            return False
    
    # Standard download using requests
    for attempt in range(max_retries):
        try:
            # Import these inside the function since they will be available
            # only after installing them in the virtual environment
            import requests
            from tqdm import tqdm
            
            print(f"Downloading {url} (Attempt {attempt+1}/{max_retries})...")
            
            # Create a session with increased timeout
            session = requests.Session()
            response = session.get(url, stream=True, timeout=60)
            response.raise_for_status()  # Raise an error for bad responses
            
            # Get file size if available
            total_size = int(response.headers.get('content-length', 0))
            
            # Setup progress bar
            progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:  # filter out keep-alive chunks
                        f.write(chunk)
                        progress_bar.update(len(chunk))
            
            progress_bar.close()
            print(f"Download completed: {output_path}")
            return True
        
        except Exception as e:
            print(f"Download failed (Attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)  # Progressive backoff
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Maximum retry attempts reached. Download failed.")
                print(f"You may need to download the file manually and place it at: {output_path}")
                print(f"Download URL: {url}")
                return False

def install_dependencies(venv_pip):
    """Install dependencies with better error handling."""
    print("Installing Python dependencies in the virtual environment...")
    
    # First try to install from requirements.txt
    try:
        print("Attempting to install from requirements.txt...")
        subprocess.run([venv_pip, "install", "-r", "requirements.txt"], check=True, capture_output=True)
        print("✓ All dependencies installed successfully from requirements.txt")
        return True
    except subprocess.CalledProcessError as e:
        print("✗ Failed to install from requirements.txt, trying individual packages...")
        if e.stderr:
            print(f"Error: {e.stderr.decode()}")
    
    # If requirements.txt fails, install packages one by one
    essential_packages = [
        "numpy",
        "torch",
        "torchvision", 
        "opencv-python",
        "pygame",
        "pyttsx3",
        "gtts",
        "face-alignment",
        "librosa",
        "scipy",
        "numba",
        "ffmpeg-python",
        "matplotlib",
        "Pillow"
    ]
    
    failed_packages = []
    
    for package in essential_packages:
        try:
            print(f"Installing {package}...")
            subprocess.run([venv_pip, "install", package], check=True, capture_output=True)
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}")
            failed_packages.append(package)
            # Try to get more info about the error
            if e.stderr:
                print(f"Error: {e.stderr.decode()}")
    
    if failed_packages:
        print(f"\nFailed to install: {', '.join(failed_packages)}")
        print("You may need to install these manually or check for compatibility issues.")
        print("Try running: pip install <package_name> manually for each failed package.")
        return False
    
    return True

def download_models_manually():
    """Provide instructions for manual download of models."""
    print("\n" + "="*80)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*80)
    print("Please download the following files manually:")
    
    print("\n1. Wav2Lip GAN model:")
    print("   URL: https://drive.google.com/file/d/15G3U08c8xsCkOqQxE38Z2XXDnPcOptNk/view?usp=share_link")
    print("   Save to: wav2lip/checkpoints/wav2lip_gan.pth")
    
    print("\n2. Face detection model:")
    print("   URL: https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth")
    print("   Alternative URL: https://iiitaphyd-my.sharepoint.com/:u:/g/personal/prajwal_k_research_iiit_ac_in/EZsy6qWuivtDnANIG73iHjIBjMSoojcIV0NULXV-yiuiIg?e=qTasa8")
    print("   Save to: wav2lip/face_detection/detection/sfd/s3fd.pth")
    
    print("\nAfter downloading, restart setup.py")
    print("="*80 + "\n")

def main():
    """Setup script to prepare the AI-Avator environment."""
    print("Setting up AI-Avator environment...")
    
    # Create virtual environment
    venv_python, venv_pip = create_venv()
    print(f"Using Python from virtual environment: {venv_python}")
    
    # Ensure required directories exist
    for directory in ['tts', 'wav2lip', 'output']:
        os.makedirs(directory, exist_ok=True)
    
    # Check if Wav2Lip is already cloned
    wav2lip_dir = Path('wav2lip')
    if not (wav2lip_dir / '.git').exists():
        print("Cloning Wav2Lip repository...")
        try:
            subprocess.run(['git', 'clone', 'https://github.com/Rudrabha/Wav2Lip.git', 'wav2lip'], check=True)
        except subprocess.CalledProcessError:
            print("Failed to clone repository. Make sure git is installed.")
            print("You can manually clone it with: git clone https://github.com/Rudrabha/Wav2Lip.git wav2lip")
    
    # Create necessary subdirectories in wav2lip
    os.makedirs('wav2lip/checkpoints', exist_ok=True)
    os.makedirs('wav2lip/face_detection/detection/sfd', exist_ok=True)
    
    # Download models - using correct URLs from the README
    download_success = True
    
    # Download Wav2Lip model if not exists
    wav2lip_model = Path('wav2lip/checkpoints/wav2lip_gan.pth')
    if not wav2lip_model.exists():
        print("Downloading Wav2Lip GAN model...")
        # Try Google Drive URL from README
        google_drive_url = "https://drive.google.com/uc?id=15G3U08c8xsCkOqQxE38Z2XXDnPcOptNk"
        if not download_file(google_drive_url, str(wav2lip_model), use_gdown=True):
            print("Failed to download Wav2Lip GAN model from Google Drive.")
            download_success = False
    
    # Download face detection model if not exists
    face_model = Path('wav2lip/face_detection/detection/sfd/s3fd.pth')
    if not face_model.exists():
        print("Downloading face detection model...")
        face_model_url = 'https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth'
        alt_face_model_url = 'https://iiitaphyd-my.sharepoint.com/:u:/g/personal/prajwal_k_research_iiit_ac_in/EZsy6qWuivtDnANIG73iHjIBjMSoojcIV0NULXV-yiuiIg?e=qTasa8'
        
        if not download_file(face_model_url, str(face_model)):
            print("Failed with primary URL, trying alternative URL...")
            if not download_file(alt_face_model_url, str(face_model)):
                print("Failed to download face detection model.")
                download_success = False
    
    # If any downloads failed, show manual download instructions
    if not download_success:
        download_models_manually()
    
    # Check for avatar.png
    avatar = Path('avatar.png')
    if not avatar.exists():
        print("WARNING: avatar.png not found in the root directory.")
        print("Please place your avatar image in the root directory before running the application.")
    
    # Check for FFmpeg
    try:
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        print("FFmpeg is installed and available.")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("WARNING: FFmpeg not found. Please install FFmpeg and make sure it's in your PATH.")
    
    # Install Python dependencies
    if not install_dependencies(venv_pip):
        print("Some dependencies failed to install. You may need to install them manually.")
    
    print("\nSetup completed! You can now run the application with:")
    if platform.system() == "Windows":
        print("venv\\Scripts\\python.exe main.py")
    else:
        print("venv/bin/python main.py")

if __name__ == "__main__":
    main()