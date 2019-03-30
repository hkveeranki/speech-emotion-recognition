from setuptools import setup


def _get_requirements():
    with open('requirements.txt') as f:
        requirements = f.read().splitlines()
    return requirements


setup(
    name='speechemotionrecognition',
    version='1.1',
    packages=['speechemotionrecognition'],
    url='http://github.com/harry-7/speech-emotion-recognition',
    license='MIT',
    install_requires=_get_requirements(),
    author='harry7',
    author_email='harry7.opensource@gmail.com',
    description='Package to do speech emotion recognition'
)
