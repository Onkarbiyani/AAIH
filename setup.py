from setuptools import setup, find_packages

setup(
    name="skin-lesion-xai",
    version="1.0.0",
    author="Onkar Biyani",
    author_email="onkarbiyani@iisc.ac.in",
    description="Explainable AI framework for skin lesion segmentation using U-Net and Grad-CAM++",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Onkarbiyani/AAIH",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    python_requires='>=3.8',
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "numpy>=1.21.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
        "grad-cam>=1.4.6"
    ],
)
