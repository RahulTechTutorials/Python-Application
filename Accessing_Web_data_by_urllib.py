import urllib.request
fileContent = urllib.request.urlopen('http://marvin.cs.uidaho.edu/Teaching/CS515/pythonTutorial.pdf')
fp = open('/Users/rahuljain/Desktop/Python/PythonPrograms/Python_Application/PythonGuidoVanRossum10.pdf','wb')
fp.writelines(fileContent)
fp.close()
