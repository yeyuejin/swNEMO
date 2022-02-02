from distutils.core import setup
setup(name='Offline observation operator',
      version='0.1',
      description='NEMO Offline observation operator',
      author='Andrew Ryan',
      author_email='andrew.ryan@metoffice.gov.uk',
      packages=['ooo',],
      scripts=['bin/ooo'],
      )
