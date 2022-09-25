from setuptools import setup

#NIRMI!
setup(
    name='NbodyIMRI',
    version='0.1.0',    
    description='A code for evolving the dynamics of intermediate mass ratio inspirals (IMRI), embedded in Dark Matter halos.',
    url='https://github.com/bradkav/NbodyIMRI',
    author='Bradley J Kavanagh',
    author_email='bradkav@gmail.com',
    license='MIT',
    packages=['NbodyIMRI'],
    install_requires=[  'h5py>=2.10.0',
                        'matplotlib>=3.3.2',
                        'numpy>=1.19.5',
                        'scipy>=1.5.3',
                        'tqdm>=4.45.0'                    
                      ]
)