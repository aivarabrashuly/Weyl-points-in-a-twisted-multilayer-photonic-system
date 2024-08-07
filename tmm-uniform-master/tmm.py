from . import backend as bd
from .fft_funs import Epsilon_fft,get_ifft
from .kbloch import Lattice_Reciprocate,Lattice_getG,Lattice_SetKs

class obj:
    def __init__(self,freq,theta,phi,verbose=1):
        '''The time harmonic convention is exp(-i omega t), speed of light = 1

        Two kinds of layers are currently supported: uniform layer,
        patterned layer from grids. Interface for patterned layer by
        direct analytic expression of Fourier series is included, but
        no examples inclded so far.

        nG: truncation order, but the actual truncation order might not be nG
        L1,L2: lattice vectors, in the list format, (x,y)

        '''
        self.freq = freq
        self.omega = 2*bd.pi*freq+0.j
        self.phi = phi
        self.theta = theta
        self.verbose = verbose
        self.Layer_N = 0  # total number of layers
      
        # the length of the following variables = number of total layers
        self.thickness_list = []
        self.id_list = []  #[type, No., No. in patterned/uniform, No. in its family] starting from 0
        # type:0 for uniform, 1 for Grids, 2 for Fourier

        self.kp_list = []                
        self.q_list = []  # eigenvalues
        self.phi_list = [] #eigenvectors

        # Uniform layer
        self.Uniform_ep_list = []
        self.Uniform_mu_list = []
        self.Uniform_xi_list = []
        self.Uniform_zeta_list = []
        self.Uniform_N = 0   
        
    def Add_LayerUniform(self,thickness,epsilon, mu, xi, zeta):
        #assert type(thickness) == float, 'thickness should be a float'

        self.id_list.append([0,self.Layer_N,self.Uniform_N])
        self.Uniform_ep_list.append(epsilon)
        self.Uniform_mu_list.append(mu)
        self.Uniform_xi_list.append(xi)
        self.Uniform_zeta_list.append(zeta)
        
        self.thickness_list.append(thickness)
        
        self.Layer_N += 1
        self.Uniform_N += 1

    def Init_Setup(self,Pscale=1.,Gmethod=0):
        '''
        Set up reciprocal lattice (Gmethod:truncation scheme, 0 for circular, 1 for rectangular)
        Pscale: scale the period
        Compute eigenvalues for uniform layers
        Initialize vectors for patterned layers
        '''
        kx0 = self.omega*bd.sin(self.theta)*bd.cos(self.phi)*bd.sqrt(self.Uniform_ep_list[0])
        ky0 = self.omega*bd.sin(self.theta)*bd.sin(self.phi)*bd.sqrt(self.Uniform_ep_list[0])

        self.Patterned_ep2_list = [None]*self.Patterned_N
        self.Patterned_epinv_list = [None]*self.Patterned_N            
        for i in range(self.Layer_N):
            if self.id_list[i][0] == 0:
                ep = self.Uniform_ep_list[self.id_list[i][2]]
                kp = MakeKPMatrix(self.omega,0,1./ep,self.kx,self.ky)
                self.kp_list.append(kp)
                
                q,phi = SolveLayerEigensystem_uniform(self.omega,self.kx,self.ky,ep)
                self.q_list.append(q)
                self.phi_list.append(phi)
            else:
                self.kp_list.append(None)
                self.q_list.append(None)
                self.phi_list.append(None)
                
    def MakeExcitationPlanewave(self,p_amp,p_phase,s_amp,s_phase,order = 0, direction = 'forward'):
        '''
        Front incidence
        '''
        self.direction = direction
        theta = self.theta
        phi = self.phi
        a0 = bd.zeros(2*self.nG,dtype=complex)
        bN = bd.zeros(2*self.nG,dtype=complex)
        if direction == 'forward':
            tmp1 = bd.zeros(2*self.nG,dtype=complex)
            tmp1[order] = 1.0
            a0 = a0 + tmp1*(-s_amp*bd.cos(theta)*bd.cos(phi)*bd.exp(1j*s_phase) \
                        -p_amp*bd.sin(phi)*bd.exp(1j*p_phase))

            tmp2 = bd.zeros(2*self.nG,dtype=complex)
            tmp2[order+self.nG] = 1.0            
            a0 = a0 + tmp2*(-s_amp*bd.cos(theta)*bd.sin(phi)*bd.exp(1j*s_phase) \
                            +p_amp*bd.cos(phi)*bd.exp(1j*p_phase))
        elif direction == 'backward':
            tmp1 = bd.zeros(2*self.nG,dtype=complex)
            tmp1[order] = 1.0
            bN = bN + tmp1*(-s_amp*bd.cos(theta)*bd.cos(phi)*bd.exp(1j*s_phase) \
                            -p_amp*bd.sin(phi)*bd.exp(1j*p_phase))

            tmp2 = bd.zeros(2*self.nG,dtype=complex)
            tmp2[order+self.nG] = 1.0
            bN = bN + tmp2*(-s_amp*bd.cos(theta)*bd.sin(phi)*bd.exp(1j*s_phase) \
                            +p_amp*bd.cos(phi)*bd.exp(1j*p_phase))
        
        self.a0 = a0
        self.bN = bN
        
            
    def RT_Solve(self,normalize = 0, byorder = 0):
        '''
        Reflection and transmission power computation
        Returns 2R and 2T, following Victor's notation
        Maybe because 2* makes S_z = 1 for H=1 in vacuum

        if normalize = 1, it will be divided by n[0]*cos(theta)
        '''
        aN, b0 = SolveExterior(self.a0,self.bN,self.q_list,self.phi_list,self.kp_list,self.thickness_list)
        fi,bi = GetZPoyntingFlux(self.a0,b0,self.omega,self.kp_list[0],self.phi_list[0],self.q_list[0],byorder=byorder)
        fe,be = GetZPoyntingFlux(aN,self.bN,self.omega,self.kp_list[-1],self.phi_list[-1],self.q_list[-1],byorder=byorder)

        if self.direction == 'forward':
            R = bd.real(-bi)
            T = bd.real(fe)
        elif self.direction == 'backward':
            R = bd.real(fe)
            T = bd.real(-bi)

        if normalize == 1:
            R = R*self.normalization
            T = T*self.normalization
        return R,T

    def GetAmplitudes_noTranslate(self,which_layer):
        '''
        returns fourier amplitude
        '''
        if which_layer == 0 :
            aN, b0 = SolveExterior(self.a0,self.bN,self.q_list,self.phi_list,self.kp_list,self.thickness_list)
            ai = self.a0
            bi = b0

        elif which_layer == self.Layer_N-1:
            aN, b0 = SolveExterior(self.a0,self.bN,self.q_list,self.phi_list,self.kp_list,self.thickness_list)
            ai = aN
            bi = self.bN

        else:
            ai, bi = SolveInterior(which_layer,self.a0,self.bN,self.q_list,self.phi_list,self.kp_list,self.thickness_list)
        return ai,bi
    
    def GetAmplitudes(self,which_layer,z_offset):
        '''
        returns fourier amplitude
        '''
        if which_layer == 0 :
            aN, b0 = SolveExterior(self.a0,self.bN,self.q_list,self.phi_list,self.kp_list,self.thickness_list)
            ai = self.a0
            bi = b0

        elif which_layer == self.Layer_N-1:
            aN, b0 = SolveExterior(self.a0,self.bN,self.q_list,self.phi_list,self.kp_list,self.thickness_list)
            ai = aN
            bi = self.bN

        else:
            ai, bi = SolveInterior(which_layer,self.a0,self.bN,self.q_list,self.phi_list,self.kp_list,self.thickness_list)

        ai, bi = TranslateAmplitudes(self.q_list[which_layer],self.thickness_list[which_layer],z_offset,ai,bi)

        return ai,bi
    
    def Solve_FieldFourier(self,which_layer,z_offset):
        '''
        returns field amplitude in fourier space: [ex,ey,ez], [hx,hy,hz]
        '''
        ai0,bi0 = self.GetAmplitudes_noTranslate(which_layer)
        #ai, bi = self.GetAmplitudes(which_layer,z_offset)

        if bd.isinstance(z_offset,float) or bd.isinstance(z_offset,int):
            zl = [z_offset]
        else:
            zl = z_offset

        eh = []
        for zoff in zl:
            ai, bi = TranslateAmplitudes(self.q_list[which_layer],self.thickness_list[which_layer],zoff,ai0,bi0)
            # hx, hy in Fourier space
            fhxy = bd.dot(self.phi_list[which_layer],ai+bi)
            fhx = fhxy[:self.nG]
            fhy = fhxy[self.nG:]

            # ex,ey in Fourier space
            tmp1 = (ai-bi)/self.omega/self.q_list[which_layer]
            tmp2 = bd.dot(self.phi_list[which_layer],tmp1)
            fexy = bd.dot(self.kp_list[which_layer],tmp2)
            fey = - fexy[:self.nG]
            fex = fexy[self.nG:]
        
            #hz in Fourier space
            fhz = (self.kx*fey - self.ky*fex)/self.omega

            #ez in Fourier space
            fez = (self.ky*fhx - self.kx*fhy)/self.omega
            if self.id_list[which_layer][0] == 0:
                fez = fez / self.Uniform_ep_list[self.id_list[which_layer][2]]
            else:
                fez = bd.dot(self.Patterned_epinv_list[self.id_list[which_layer][2]],fez)
            eh.append([[fex,fey,fez],[fhx,fhy,fhz]])
        return eh

    def Solve_FieldOnGrid(self,which_layer,z_offset,Nxy=None):
        # Nxy = [Nx,Ny], if not supplied, will use the number in patterned layer
        # if single z_offset:  output [[ex,ey,ez],[hx,hy,hz]]
        # if z_offset is list: output [[[ex1,ey1,ez1],[hx1,hy1,hz1]],  [[ex2,ey2,ez2],[hx2,hy2,hz2]] ...]

        if bd.isinstance(Nxy,type(None)):
            Nxy = self.GridLayer_Nxy_list[self.id_list[which_layer][3]]
        Nx = Nxy[0]
        Ny = Nxy[1]

        # e,h in Fourier space
        fehl = self.Solve_FieldFourier(which_layer,z_offset)

        eh = []
        for feh in fehl:
            fe = feh[0]
            fh = feh[1]
            ex = get_ifft(Nx,Ny,fe[0],self.G)
            ey = get_ifft(Nx,Ny,fe[1],self.G)
            ez = get_ifft(Nx,Ny,fe[2],self.G)

            hx = get_ifft(Nx,Ny,fh[0],self.G)
            hy = get_ifft(Nx,Ny,fh[1],self.G)
            hz = get_ifft(Nx,Ny,fh[2],self.G)
            eh.append([[ex,ey,ez],[hx,hy,hz]])
        if bd.isinstance(z_offset,float) or bd.isinstance(z_offset,int):
            eh = eh[0]
        return eh

    def Volume_integral(self,which_layer,Mx,My,Mz,normalize=0):
        '''Mxyz is convolution matrix.
        This function computes 1/A\int_V Mx|Ex|^2+My|Ey|^2+Mz|Ez|^2
        To be consistent with Poynting vector defintion here, the absorbed power will be just omega*output
        '''
        kp = self.kp_list[which_layer]
        q = self.q_list[which_layer]
        phi = self.phi_list[which_layer]

        if self.id_list[which_layer][0] == 0:
            epinv = 1. / self.Uniform_ep_list[self.id_list[which_layer][2]]
        else:
            epinv = self.Patterned_epinv_list[self.id_list[which_layer][2]]

        # un-translated amplitdue
        ai, bi = SolveInterior(which_layer,self.a0,self.bN,self.q_list,self.phi_list,self.kp_list,self.thickness_list)
        ab = bd.hstack((ai,bi))
        abMatrix = bd.outer(ab,bd.conj(ab))
        
        Mt = Matrix_zintegral(q,self.thickness_list[which_layer])
        # overall
        abM = abMatrix * Mt

        # F matrix
        Faxy = bd.dot(bd.dot(kp,phi), bd.diag(1./self.omega/q))
        Faz1 = 1./self.omega*bd.dot(epinv,bd.diag(self.ky))
        Faz2 = -1./self.omega*bd.dot(epinv,bd.diag(self.kx))
        Faz = bd.dot(bd.hstack((Faz1,Faz2)),phi)

        tmp1 = bd.vstack((Faxy,Faz))
        tmp2 = bd.vstack((-Faxy,Faz))
        F = bd.hstack((tmp1,tmp2))

        # consider Mtotal
        Mzeros = bd.zeros_like(Mx)
        Mtotal = bd.vstack((bd.hstack((Mx,Mzeros,Mzeros)),\
                            bd.hstack((Mzeros,My,Mzeros)),\
                            bd.hstack((Mzeros,Mzeros,Mz))))

        # integral = Tr[ abMatrix * F^\dagger *  Matconv *F ] 
        tmp = bd.dot(bd.dot(bd.conj(bd.transpose(F)),Mtotal),F)
        val = bd.trace(bd.dot(abM,tmp))

        if normalize == 1:
            val = val*self.normalization
        return val
        
    def Solve_ZStressTensorIntegral(self,which_layer):
        '''
        returns 2F_x,2F_y,2F_z, integrated over z-plane
        '''
        z_offset = 0.
        eh = self.Solve_FieldFourier(which_layer,z_offset)
        e = eh[0][0]
        h = eh[0][1]
        ex = e[0]
        ey = e[1]
        ez = e[2]

        hx = h[0]
        hy = h[1]
        hz = h[2]

        # compute D = epsilon E
        ## Dz = epsilon_z E_z = (ky*hx - kx*hy)/self.omega
        dz = (self.ky*hx - self.kx*hy)/self.omega

        ## Dxy = epsilon2 * Exy
        if self.id_list[which_layer][0] == 0:
            dx = ex * self.Uniform_ep_list[self.id_list[which_layer][2]]
            dy = ey * self.Uniform_ep_list[self.id_list[which_layer][2]]
        else:
            exy = bd.hstack((-ey,ex))
            dxy = bd.dot(self.Patterned_ep2_list[self.id_list[which_layer][2]],exy)
            dx = dxy[self.nG:]
            dy = -dxy[:self.nG]

        Tx = bd.sum(ex*bd.conj(dz)+hx*bd.conj(hz))
        Ty = bd.sum(ey*bd.conj(dz)+hy*bd.conj(hz))
        Tz = 0.5*bd.sum(ez*bd.conj(dz)+hz*bd.conj(hz)-ey*bd.conj(dy)-ex*bd.conj(dx)-bd.abs(hx)**2-bd.abs(hy)**2)

        Tx = bd.real(Tx)
        Ty = bd.real(Ty)
        Tz = bd.real(Tz)

        return Tx,Ty,Tz

def ComputeNu(omega, which_layer, kx0, ky0, Uniform_ep_list, Uniform_mu_list, Uniform_xi_list, Uniform_zeta_list):
    det = Uniform_ep_list[which_layer][2][2]*Uniform_mu_list[which_layer][2][2]-Uniform_xi_list[which_layer][2][2]*Uniform_zeta_list[which_layer][2][2]
    nuzxee = - (Uniform_mu_list[which_layer][2][2]*Uniform_ep_list[which_layer][2][0]-Uniform_xi_list[which_layer][2][2]*(Uniform_zeta_list[which_layer][2][0]+ky0/omega))/det
    nuzyee = - (Uniform_mu_list[which_layer][2][2]*Uniform_ep_list[which_layer][2][1]-Uniform_xi_list[which_layer][2][2]*(Uniform_zeta_list[which_layer][2][1]-kx0/omega))/det
    nuzxeh = (Uniform_xi_list[which_layer][2][2]*Uniform_mu_list[which_layer][2][0]-Uniform_mu_list[which_layer][2][2]*(Uniform_xi_list[which_layer][2][0]-ky0/omega))/det
    nuzyeh = (Uniform_xi_list[which_layer][2][2]*Uniform_mu_list[which_layer][2][1]-Uniform_mu_list[which_layer][2][2]*(Uniform_xi_list[which_layer][2][1]+kx0/omega))/det
    nuzxhe = (Uniform_zeta_list[which_layer][2][2]*Uniform_ep_list[which_layer][2][0]-Uniform_ep_list[which_layer][2][2]*(Uniform_zeta_list[which_layer][2][0]+ky0/omega))/det
    nuzyhe = (Uniform_zeta_list[which_layer][2][2]*Uniform_ep_list[which_layer][2][1]-Uniform_ep_list[which_layer][2][2]*(Uniform_zeta_list[which_layer][2][1]-kx0/omega))/det
    nuzxhh = - (Uniform_ep_list[which_layer][2][2]*Uniform_mu_list[which_layer][2][0]-Uniform_zeta_list[which_layer][2][2]*(Uniform_xi_list[which_layer][2][0]-ky0/omega))/det
    nuzyhh = - (Uniform_ep_list[which_layer][2][2]*Uniform_mu_list[which_layer][2][1]-Uniform_zeta_list[which_layer][2][2]*(Uniform_xi_list[which_layer][2][1]+kx0/omega))/det
    return nuzxee, nuzyee, nuzxeh, nuzyeh, nuzxhe, nuzyhe, nuzxhh, nuzyhh
    
def MakePMatrix(omega, which_layer, kx0, ky0, Uniform_ep_list, Uniform_mu_list, Uniform_xi_list, Uniform_zeta_list):
    J = bd.ones((4,4))
    
    P = bd.zeros((4,4))
    
    nuzxee, nuzyee, nuzxeh, nuzyeh, nuzxhe, nuzyhe, nuzxhh, nuzyhh = ComputeNu(omega, which_layer, kx0, ky0, Uniform_ep_list, Uniform_mu_list, Uniform_xi_list, Uniform_zeta_list)
    
    temp0 = bd.array([[Uniform_zeta_list[which_layer][1][0], Uniform_zeta_list[which_layer][1][1], Uniform_mu_list[which_layer][1][0], Uniform_mu_list[which_layer][1][1]], [-Uniform_zeta_list[which_layer][0][0], -Uniform_zeta_list[which_layer][0][1], -Uniform_mu_list[which_layer][0][0], -Uniform_mu_list[which_layer][0][1]], [-Uniform_ep_list[which_layer][1][0], Uniform_ep_list[which_layer][1][1], -Uniform_xi_list[which_layer][1][0], -Uniform_xi_list[which_layer][1][1]], [Uniform_ep_list[which_layer][0][0], Uniform_ep_list[which_layer][0][1], Uniform_xi_list[which_layer][0][0], Uniform_xi_list[which_layer][0][1]]])
    temp1 = bd.diag((Uniform_zeta_list[which_layer][1][2]+kx0/omega, -Uniform_zeta_list[which_layer][0][2]+ky0/omega, -Uniform_ep_list[which_layer][1][2], Uniform_ep_list[which_layer][0][2]))
    temp2 = bd.diag((nuzxee, nuzyee, nuzxeh, nuzyeh))
    
    temp3 = bd.diag((Uniform_mu_list[which_layer][1][2], -Uniform_mu_list[which_layer][0][2], -Uniform_xi_list[which_layer][1][2]+kx0/omega, Uniform_xi_list[which_layer][0][2]+ky0/omega))
    temp4 = bd.diag((nuzxhe, nuzyhe, nuzxhh, nuzyhh))
    
    P = omega*(temp0 + bd.dot(bd.dot(temp1, J), temp2) + bd.dot(bd.dot(temp3, J), temp4))
    
    return P

def SolveEigensystem(omega, which_layer, kx0, ky0, Uniform_ep_list, Uniform_mu_list, Uniform_xi_list, Uniform_zeta_list):
    
    P = MakePMatrix(omega, which_layer, kx0, ky0, Uniform_ep_list, Uniform_mu_list, Uniform_xi_list, Uniform_zeta_list)
    
    w, v = bd.linalg.eig(P)

    
    return w, v

def MakeMMatrix(omega, which_layer, kx0, ky0, Uniform_ep_list, Uniform_mu_list, Uniform_xi_list, Uniform_zeta_list, thickness):
    
    W, V = SolveEigensystem(omega, which_layer, kx0, ky0, Uniform_ep_list, Uniform_mu_list, Uniform_xi_list, Uniform_zeta_list)
    
    expG = bd.zeros((4, 4))
    
    for i in range(4):
        expG[i, i] = bd.exp(1j*W[i]*thickness)
       
    
    M  = V @ W @ bd.linalg.inv(V) 
    
    return M
    