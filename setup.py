

from setuptools import setup, find_packages

setup(
    name="Actividad_3 y Final",
    version="0.0.3",
    author="Mauricio Alejandro Gonzalez Guerra",
    author_email="mauricio.gonzalez@est.iudigital.edu.co",
    description="Evidencia de aprendizaje Numero 3 y Final",
    py_modules=["Actividad_3", "Actividad_Final"],
    install_requires=[
        "kagglehub[pandas-datasets]>=0.3.8",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.2",
        "pandas",
        "numpy",
        "openpyxl",
        "requests"
    ]
    
    
)


