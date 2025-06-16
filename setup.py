from setuptools import find_packages,setup
from typing import List
def get_requirements(file_path:str)->List[str]:
    '''
    This is over here converts all the packages in requirements.txt into list and sends to install requirements so that it will be flexible 
    '''
    hypen="-e ."
    requirements=[]
    with open(file_path) as file:
        requirements=file.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if hypen in requirements:
            requirements.remove(hypen)
    return requirements
    

setup(
    name="ML Project",
    version='0.0.1',
    author="Sai Mani",
    author_email="macherlasaimani@gmail.com",
    packages=find_packages(),#How the find packages will find out all the packages that are included in the ML project is to see the source folder 
    install_requires=get_requirements('requirements.txt')
    
    
)