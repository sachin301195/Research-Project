import getpass
from pathlib import Path
import platform

if getpass.getuser() == 'sachin':
    print('I got you Sachin')
else:
    print('Try something else')
print(platform.system())