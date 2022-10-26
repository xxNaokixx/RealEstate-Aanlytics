import pathlib
import string
import tempfile

s = """


Hi $name,

$contents

havea Googday
"""


#with open("test.txt", "w") as f:
   # f.write(s)

#with open("test.txt", "r") as f:
    #print(f.read())
    #while True:
     #   chunk = 2
      #  line = f.read(chunk)
       # print(line)
        #if not line:
         #   break

    #print(f.tell())
    #print(f.read(1))
    #f.seek(2)
    #print(f.read(1))
    #f.seek(15)
    #print(f.rea


t = string.Template(s)
contents = t.substitute(name = "Mike", contents = "How are you ?")   
print(contents)


#with open("test.txt", "w+") as f:
 #   f.write(s)
  #  f.seek(0)
   # print(f.read())


import csv

with open("test.csv", "w") as csv_file:
    fieldnames = ["Name", "Count"]
    writer = csv.DictWriter(csv_file, fieldnames = fieldnames)
    writer.writeheader()
    writer.writerow({"Name": "A", "Count": 1})
    writer.writerow({"Name": "B", "Count": 2})

with open("test.csv", "r") as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        print(row["Name"], row["Count"])


import os

print(os.path.exists("test.txt"))
print(os.path.isfile("test.txt"))
print(os.path.isdir("test.txt"))

#os.rename("test.txt", "renamed.txt")

#os.mkdir("test.dir")
#os.rmdir("test.dir")

pathlib.Path("empty.txt").touch()

import tarfile

import zipfile

import tempfile

#with tempfile.TemporaryFile(mode = "w") as t:
 #   t.write("Hello")
  #  t.seek(0)
   # print(t.read())

#with tempfile.NamedTemporaryFile(delete=False) as t:
 #   print(t.name)
  #  with open(t.name, "w+") as f:
   #     f.write("test")
    #    f.seek(0)
     #   print(f.read())
#

import subprocess
subprocess.run(["ls", "-al"])

import datetime
now = datetime.datetime.now()

print(now)
print(now.isoformat())

today = datetime.datetime.today()