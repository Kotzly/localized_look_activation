import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
     name='LocalizedLook',
     version='0.1',
     author="Kotzly",
     author_email="paullo.augusto@hotmail.com",
     description="Localized activation for sparse patterns discovering.",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/Kotzly/localized_look_activation",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "Operating System :: OS Independent",
     ],
     python_requires='>=3.7.6',
     install_requires=required,
 )
