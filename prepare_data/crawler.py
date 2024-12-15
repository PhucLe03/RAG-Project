import os
import subprocess
import shutil

def download_folders_git(repo_url, tag, folders, destination):
    """
    Downloads specific folders from a GitHub repository using git sparse-checkout.

    Args:
        repo_url (str): The URL of the GitHub repository.
        tag (str): The tag containing the desired folders.
        folders (list): List of folder paths to download.
        destination (str): Destination directory for downloaded content.
    """
    # Create destination directory
    os.makedirs(destination, exist_ok=True)

    # Initialize a bare git repository
    print(f"Cloning repository: {repo_url}")
    subprocess.run(["git", "clone", "--no-checkout", repo_url, destination], check=True)

    # Navigate to the destination directory
    os.chdir(destination)

    # Set up sparse-checkout
    print("Initializing sparse-checkout...")
    subprocess.run(["git", "sparse-checkout", "init", "--cone"], check=True)

    # Add all folders to sparse-checkout in one command
    print(f"Setting sparse-checkout folders: {folders}")
    subprocess.run(["git", "sparse-checkout", "set", *folders], check=True)

    # Checkout the specified tag
    print(f"Checking out tag: {tag}")
    subprocess.run(["git", "checkout", f"tags/{tag}"], check=True)

    # Cleanup: Remove all non-selected folders/files except .git
    print("Cleaning up root directory...")
    for item in os.listdir("."):
        # Skip system directories like .git and the folders we want to keep
        if item not in folders and item != ".git":
            try:
                if os.path.isfile(item) or os.path.islink(item):
                    os.unlink(item)  # Remove files and symbolic links
                elif os.path.isdir(item):
                    shutil.rmtree(item)  # Remove directories
            except PermissionError as e:
                print(f"Skipping protected file or folder: {item}. Reason: {e}")

    print("Download complete.")

if __name__ == "__main__":
    # Example usage
    repo_url = "https://github.com/vercel/next.js"
    tag = "v15.1.1-canary.1"  # Replace with the desired tag
    folders = ["docs", "errors"]
    destination = "./v15.1.1-canary.1"

    download_folders_git(repo_url, tag, folders, destination)
