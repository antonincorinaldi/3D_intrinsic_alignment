def simulation(mu_B, mu_C, sigma_B, sigma_C, r, nb_simu, p_axis='y', A=1,e_bins=np.linspace(0,1,100)):

    # Random (gaussian) axis lengths B,C
    BC = np.random.multivariate_normal(mean=[mu_B, mu_C], cov=[[sigma_B**2, r], [r, sigma_C**2]], size=nb_simu)
    mask = (BC[:,0]/A<=1) & (BC[:,1]/ A<=1) & (BC[:,0] >= BC[:,1]) & (BC[:,0]>0) & (BC[:,1]>0)
    BC2 = BC[mask]

    B=BC2[:,0] ; C=BC2[:,1]
    A=np.ones(len(BC2)) # A fixed to 1

    nb_simu=len(BC2)


    # Random orientation angles
    rand_quat = np.random.randn(nb_simu,4) ; rand_quat /= np.linalg.norm(rand_quat, axis=1, keepdims=True)
    rotation = Rotation.from_quat(rand_quat) ; euler_angles = rotation.as_euler('ZYX', degrees=True)
    euler_angles_rad = euler_angles*np.pi/180
    psi = euler_angles_rad[:,0]; theta = euler_angles_rad[:,1] ; phi = euler_angles_rad[:,2]

    
    # Rotation matrix
    D = np.zeros((nb_simu, 3, 3))

    D[:, 0, 0] = np.cos(theta) * np.cos(psi)
    D[:, 0, 1] = -np.cos(phi) * np.sin(psi) + np.sin(phi) * np.sin(theta) * np.cos(psi)
    D[:, 0, 2] = np.sin(phi) * np.sin(psi) + np.cos(phi) * np.sin(theta) * np.cos(psi)

    D[:, 1, 0] = np.cos(theta) * np.sin(psi)
    D[:, 1, 1] = np.cos(phi) * np.cos(psi) + np.sin(phi) * np.sin(theta) * np.sin(psi)
    D[:, 1, 2] = -np.sin(phi) * np.cos(psi) + np.cos(phi) * np.sin(theta) * np.sin(psi)

    D[:, 2, 0] = -np.sin(theta)
    D[:, 2, 1] = np.sin(phi) * np.cos(theta)
    D[:, 2, 2] = np.cos(phi) * np.cos(theta)



    #Eigenvectors X-Y-Z
    evc0 = np.asarray([[1,0,0],[0,1,0],[0,0,1]])
    evcl = np.einsum('ijk,kl->ijl', D, evc0)


    #Eigenvalues Maj-Mid-Min
    evls = np.array([A,B,C])**2 ; evls=np.transpose(evls)


    # Projection 3D => 2D
    if p_axis=='x':
        K = np.sum(evcl[:,:,0][:,:,None]*(evcl/evls[:,None]), axis=1)
        r = evcl[:,:,2] - evcl[:,:,0] * K[:,2][:,None] / K[:,0][:,None]
        s = evcl[:,:,1] - evcl[:,:,0] * K[:,1][:,None] / K[:,0][:,None] 

    if p_axis=='y':
        K = np.sum(evcl[:,:,1][:,:,None] * (evcl/evls[:,None]), axis=1)
        r = evcl[:,:,0] - evcl[:,:,1] * K[:,0][:,None] / K[:,1][:,None]
        s = evcl[:,:,2] - evcl[:,:,1] * K[:,2][:,None] / K[:,1][:,None]

    A1 = np.sum(r**2 / evls, axis=1)
    B1 = np.sum(2*r*s / evls, axis=1)
    C1 = np.sum(s**2 / evls, axis=1)

    theta = np.pi / 2 + np.arctan2(B1,A1-C1)/2
    a_p = 1/np.sqrt((A1+C1)/2 + (A1-C1)/(2*np.cos(2*theta)))
    b_p = 1/np.sqrt(A1+C1-(1/a_p**2))

    def e_complex(a,b,r):
        abs_e = (1-(b/a)) / (1+(b/a))
        e1 = abs_e*np.cos(2*r)
        e2 = abs_e*np.sin(2*r)
        return e1, e2

    e1, e2 = e_complex(a_p, b_p, theta) ; e = [e1,e2] ; e=np.array(e)

    e_counts,_ = np.histogram(np.sqrt(e.T[:,0]**2+e.T[:,1]**2),bins=e_bins)
    
    return e, e_counts
