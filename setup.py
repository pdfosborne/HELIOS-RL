from distutils.core import setup

setup(
    name='HELIOS-RL-Test',
    version='0.3.4',
    packages=[
        'helios_rl', 
        'helios_rl.adapters', 
        'helios_rl.agents', 
        'helios_rl.encoders', 
        'helios_rl.environment_setup', 
        'helios_rl.evaluation', 
        'helios_rl.experiments'],
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