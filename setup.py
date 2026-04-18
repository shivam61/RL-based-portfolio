from setuptools import setup, find_packages

setup(
    name="rl-portfolio",
    version="1.0.0",
    description="RL-based hierarchical portfolio management for Indian equities",
    author="Quant Research",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=open("requirements.txt").read().splitlines(),
    entry_points={
        "console_scripts": [
            "rlp-download=scripts.download_data:main",
            "rlp-backtest=scripts.run_backtest:main",
            "rlp-rl=scripts.run_rl:main",
        ]
    },
)
