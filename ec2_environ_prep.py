import os


"""
1. Installs python libraraies
2. Installs aws cli (if needed)
3. Sets up AWS credentials with aws configure (if needed)
4. Git clone project repo
"""


def install_dependancies(py_libraries, aws_):
    #Install python libraries
    for py_lib in py_libraries:
        os.system("pip install {}".format(py_lib))

    #Install AWS command Line Interface
    if aws_ :
        os.system("pip install awscli")
    #Create subfolders
    os.system ("mkdir music_downloads")
    os.system ("mkdir error_log")
    os.system ("touch error_log/error.txt")


def git_clone_repo(repo_url):

    os.system("cd")
    try:
        os.system("git clone {}".format(repo_url))
    except:
        print "Error occured, please check repo link"


def set_aws_keys():
    os.system("aws configure")

if __name__ == "__main__":
    py_libraries = ["pause", "StringIO"]

    aws = str(raw_input("Do you need to access AWS services? (y/n)"))

    if aws == "y":
            py_libraries.append("boto")
            print "Configure AWS access keys.. \n"
            set_aws_keys()
            print "If you are using boto for AWS access, you may need to configure and re-run .bashrc "
            install_dependancies(py_libraries, aws_=True)


    else:
        install_dependancies(py_libraries, aws_=False)
        git_clone_repo(raw_input("Enter git repo URL: "))

    print "\n All set!"
