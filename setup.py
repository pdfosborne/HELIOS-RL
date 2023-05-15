from distutils.core import setup

setup(
    name='HELIOS-RL-TEST',
    version='0.3.3',
    packages=['heliosRL'],
    url='https://github.com/pdfosborne/HELIOS-RL',
    license='GNU Public License v3',
    author='Philip Osborne',
    author_email='pdfosborne@gmail.com',
    description='Applying the HELIOS architecture to Reinforcement Learning problems.',
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scipy',
        'torch',
        'sentence-transformers'
    ]
)