from setuptools import setup, find_packages

setup(
    name="moodmentor",
    version="4.0.0",
    author="Ritesh Tiwari",
    description="Mood Mentor — Enterprise AI Psychological Wellness Platform",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "streamlit>=1.28.0",
        "plotly>=5.17.0",
        "pandas>=2.0.0",
        "pytest>=7.4.0"
    ],
    entry_points={
        "console_scripts": [
            "moodmentor=moodmentor.cli:main",
        ],
    },
    python_requires=">=3.9",
)
