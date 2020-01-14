from distutils.extension import Extension
from Cython.Build import cythonize
import setuptools


extensions = [Extension("magnet/*", ["magnet/*.pyx"])]
setuptools.setup(
    name="magnet-learn",
    setup_requires=["cython", "keras", "tqdm", "networkx"],
    ext_modules=cythonize(extensions),
    version="0.0.1",
    author="Maixent Chenebaux",
    author_email="max.chbx@gmail.com",
    description="MAnifold learning form weighted Graphs and NETworks",
    url="https://github.com/kerighan/magnet",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"])
