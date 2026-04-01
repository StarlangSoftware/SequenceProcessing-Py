from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="nlptoolkit_sequenceprocessing",
    version="1.0.1",
    packages=['SequenceProcessing', 'SequenceProcessing.Classification', 'SequenceProcessing.Functions',
              'SequenceProcessing.Parameters', 'SequenceProcessing.Sequence'],
    url="https://github.com/StarlangSoftware/SequenceProcessing-Py",
    author="olcaytaner",
    author_email="olcay.yildiz@ozyegin.edu.tr",
    description="Sequence Processing library",
    install_requires=[
        "NlpToolkit-Math",
        "NlpToolkit-ComputationalGraph",
        "NlpToolkit-WordToVec",
        "NlpToolkit-Corpus",
        "NlpToolkit-Classification",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
)