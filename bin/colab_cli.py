from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import argparse
from os.path import abspath, dirname, join
import os
def parse_args(*args, **kwargs):
    parser = argparse.ArgumentParser(description="Upload API for ECE 542 Final Project")
    parser.add_argument('--master', action="store_true", dest='master')
    parser.add_argument('--branch', action="store_true", dest="branch")
    parser.add_argument('--down', action="store_true", dest='down')
    parser.add_argument('--up', action='store_false', dest='down')
    return parser.parse_args()

def set_path():
    abs = abspath(__file__)
    dname = dirname(abs)
    os.chdir(dname)

def auth():
    gauth = GoogleAuth()
    gauth.LocalWebserverAuth()

    return  GoogleDrive(gauth)

def upload_branch():
    conf = {'title': "ECE542FinalProjectBranch.ipynb"}
    upload_colab(conf)

def upload_master():
    conf = {
        'title': "ECE542FinalProject.ipynb",
        'parents': [{'id':"1SIPoJ2CtVKLmcXn78yDacNphsD6gSMJe"}]
    }
    upload_colab(conf)
    

def upload_colab(config):
    set_path()
    drive = auth()
    
    file1 = drive.CreateFile(config)
    
    path = join(dirname(dirname(__file__)), '..')
    path = join(path, 'ECE542FinalProject.ipynb')

    file1.SetContentFile(str(path))
    file1.Upload()

def download_colab(config):
    set_path()
    drive = auth()

    file1 = drive.CreateFile(config)

    path = join(dirname(dirname(__file__)), '..')
    path = join(path, 'ECE542FinalProject.ipynb')
    file1.GetContentFile(str(path)) #overwrites existing content



def download_branch():
    #will depend on your drive stuff will need modifications
    pass

def download_master():
    conf = {'id': "1mkJazb1EA_HvCIjQryyD1R23mKFObL09#scrollTo=dNtCvd6KxZO0"}
    download_colab(config)

if __name__ == "__main__" :
    results = parse_args()
    
    if(results.down):
        if(results.branch):
            download_branch()
        elif(results.master):
            download_master()
    else:
        if(results.master):
            upload_master()
        elif(results.branch):
            upload_branch()