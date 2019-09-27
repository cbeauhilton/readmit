from setuptools import setup
import versioneer

requirements = [
    # package requirements go here
]

setup(
    name='cbhHelpers',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Beau's helpers",
    author="Beau Hilton",
    author_email='cbeauhilton@gmail.com',
    url='https://github.com/cbeauhilton/cbhHelpers',
    packages=['cbh'],
    entry_points={
        'console_scripts': [
            'cbh=cbh.cli:cli'
        ]
    },
    install_requires=requirements,
    keywords='cbhHelpers',
    classifiers=[
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
    ]
)
