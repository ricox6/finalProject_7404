from setuptools import setup, find_packages

setup(
    name="jumanji-pacman",
    version="0.1",
    packages=find_packages(where="."),  # 自动发现所有子包
    package_dir={"": "."},              # 根目录为当前文件夹
    install_requires=["jax", "jumanji"],  # 添加依赖项
)