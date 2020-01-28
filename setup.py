# !/usr/bin/env python

from distutils.core import setup


def parse_requirements_file(filename):
    with open(filename) as fid:
        requires = [l.strip() for l in fid.readlines() if l]

    return requires

INSTALL_REQUIRES = parse_requirements_file('requirements.txt')
print(INSTALL_REQUIRES)

setup(
    name='segmentify',
    packages=[],
    version='0.0.0',
    description='Python image segmentation plugin.',
    maintainer='Nicholas Sofroniew',
    maintainer_email='sofroniewn@gmail.com',
    license='BSD 3-Clause',
    url='https://github.com/transformify-plugins/segmentify',
    keywords=['transformify-plugins', 'transformify', 'segmentify', ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Plugins',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Utilities',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
    ],
    install_requires=INSTALL_REQUIRES,
)
