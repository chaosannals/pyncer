import setuptools
from ran import fs

setuptools.setup(
    name='pyncer',
    version='0.0.1',
    description='yet a captcha library',
    long_description=fs.load('readme.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/chenshenchao/pyncer',
    keywords='pyncer captcha',
    license='MIT',
    author='chenshenchao',
    author_email='chenshenchao@outlook.com',
    platforms='any',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    packages=setuptools.find_packages(),
    install_requires=[
        'ran>=0.0.5',
        'numpy>=1.21.1',
        'pillow>=8.3.1',
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'torchaudio>=0.9.0',
        'torchnet>=0.0.4',
    ]
)
