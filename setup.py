from setuptools import setup, find_packages
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='FDLPIPS',
    version='1.0',
    description='Frequency Distribution Loss (FDL) + LPIPS',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=['FDLPIPS'],
    author='Zhangkai Ni, Juncheng Wu, Zian Wang',
    author_email='zkni@tongji.edu.cn',
    install_requires=["torch>=1.0", "torchvision"],
    url='https://github.com/spacepxl/FDLPIPS',
    keywords = ['pytorch', 'loss', 'image transformation','misalignment'],
    platforms = "python",
    license='MIT',
)