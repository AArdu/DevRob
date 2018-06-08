import matlab.engine as MATLAB
import os
import time

eng = MATLAB.start_matlab()
eng.load("./all_data/trainedRBF.mat")

while True:
    while os.stat("./all_data/centers_read.txt").st_size == 0:
        time.sleep(1)

    time.sleep(1)
    with open("centers_read.txt", "r+") as c_file:
        centers = c_file.read()
        c_file.write()

    open("centers_read.txt", 'w').close()

    eng.workspace['centers'] = centers
    eng.eval("new_joints = sim(net, centers);", nargout=0)

    new_joints = eng.workspace['new_joints']
    with open("joints_read.txt", "w+") as j_file:
        j_file.write(new_joints)
