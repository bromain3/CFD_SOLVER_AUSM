def log(path,file,nb_iter,dateini,date_final,simulation_time,converged):
    """
    Saves the log of the simulation in the chosen file
    :param path: path to the save folder (str)
    :param file: name of the save file (str)
    :param nb_iter: number of iterations of the simulation (int)
    :param dateini: date of the beginning of the simulation (str)
    :param date_final: date of the end of the simulation (str)
    :param simulation_time: length of the simulation in seconds (float)
    :param converged: flag to indicate if the simulation converged (bool)
    """
    with open(path+'/'+file, 'w') as fout:
        fout.write('Number of iterations: '+str(nb_iter)+'\n')
        fout.write('Beginning of simulation: '+dateini+'\n')
        fout.write('End of simulation: ' + date_final + '\n')
        fout.write('Time of simulation (seconds): ' + str(simulation_time) + '\n')
        fout.write('Simulation convergence: ' + str(converged) + '\n')


def residuals(path,file,Cr,Xr,Yr,Er):
    """
    Saves the residuals in the chosen file
    :param path: path to the save folder (str)
    :param file: name of the save file (str)
    :param Cr: continuity residual (list)
    :param Xr: X-momentum residual (list)
    :param Yr: Y-momentum residual (list)
    :param Er: energy residual (list)
    """
    with open(path+'/'+file, "w") as fout:
        fout.write("continuity residual;x momentum residual;y momentum residual;energy residual"+'\n')
        for i in range(len(Cr)):
            fout.write(str(Cr[i])+';'+str(Xr[i])+';'+str(Yr[i])+';'+str(Er[i])+'\n')

def computational_time(path,file,T):
    """
    Saves the simulation time in the chosen file
    :param path: path to the save folder (str)
    :param file: name of the save file (str)
    :param T: list of simulation time of each time steps (list)
    """
    with open(path+'/'+file, "w") as fout:
        for i in range(len(T)):
            fout.write(str(T[i])+'\n')

def states(path,file,X,nx,ny):
    """
    Saves the state vectors and the data required for reconstruction in the chosen file
    :param path: path to the save folder (str)
    :param file: name of the save file (str)
    :param X: matrix of state vectors (3D-arrays)
    :param nx: dimension of the second axis of the array X (int)
    :param ny: dimension of the first axis of the array X (int)
    """
    rho = X[:,:,0].flatten()
    rho_u = X[:, :, 1].flatten()
    rho_v = X[:, :, 2].flatten()
    rho_et = X[:, :, 3].flatten()
    with open(path+'/'+file, "w") as fout:
        fout.write('nx:'+str(nx)+' ny:'+str(ny)+'\n')
        fout.write("rho;rho*u;rho*v;rho*et"+'\n')
        for i in range(len(rho)):
            fout.write(str(rho[i])+';'+str(rho_u[i])+';'+str(rho_v[i])+';'+str(rho_et[i])+'\n')