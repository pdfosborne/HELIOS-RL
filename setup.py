from distutils.core import setup

setup(
    name='helios',
    version='0.1.1',
    packages=[
        'helios', 
        'helios.adapters', 
        'helios.agents', 
        'helios.encoders', 
        'helios.environment_setup', 
        'helios.evaluation', 
        'helios.experiments'],
    url='',
    license='',
    author='Philip Osborne',
    author_email='XXX',
    description='XXX',
    install_requires=[
        'numpy',
        'pandas',
        'torch',
        'sentence_transformers',
        'matplotlib'
    ]
)