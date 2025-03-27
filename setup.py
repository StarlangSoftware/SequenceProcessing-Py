from setuptools import setup

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name='NlpToolkit-SequenceProcessing',
    version='1.0.0',
    packages=['SequenceProcessing', 'SequenceProcessing.Classification', 'SequenceProcessing.Initializer',
              'SequenceProcessing.Sequence'],
    url='https://github.com/StarlangSoftware/SequenceProcessing-Py',
    license='',
    author='olcaytaner',
    author_email='olcay.yildiz@ozyegin.edu.tr',
    description='Sequence Processing library',
    install_requires=['NlpToolkit-Math', 'NlpToolkit-WordToVec', 'NlpToolkit-Corpus', 'NlpToolkit-Classification'],
    long_description=long_description,
    long_description_content_type='text/markdown'
)
