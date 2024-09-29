import os
import shutil
from setuptools import setup, find_packages
from setuptools.command.install import install

class PostInstallCommand(install):
    def run(self):
        print("Running custom installation process.")
        # Run the standard install first
        install.run(self)

        # Define the path to your modified block.py file
        # Assuming it's in the same directory as setup.py
        source_file = os.path.join(os.path.dirname(__file__), 'modified_block.py')

        # Find the installation directory
        install_dir = self.install_lib
        
        # Define the target path
        target_file = os.path.join(install_dir, 'site-packages', 'awq', 'modules', 'fused', 'block.py')

        # Replace the file
        if os.path.exists(source_file):
            os.makedirs(os.path.dirname(target_file), exist_ok=True)
            shutil.copy2(source_file, target_file)
            print(f"Replaced {target_file} with custom version.")
        else:
            print(f"Warning: Custom file {source_file} not found. Using default version.")

setup(
    name='llama_feature_analysis',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',  # Specify minimum version, let user's environment decide
        'autoawq',
        'numpy',
        'scipy',
        'matplotlib',
        'tqdm',
        'pandas',  # Useful for data manipulation and analysis
        'seaborn',  # For enhanced visualizations
        'scikit-learn',  # Might be useful for analysis
    ],
    cmdclass={
        'install': PostInstallCommand,
    },
)


if __name__ == '__main__':
    print("Custom installation process started.")
    # You can add more custom logic here if needed