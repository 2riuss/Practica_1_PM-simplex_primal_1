import numpy as np

class Pe:
    
    def __init__(self):
        self.c = -1
        self.A = -1
        self.b = -1
        self.m, self.n = -1, -1
        self.eps = 10e-5
    
    def read(self):
        print("m, n: ")
        self.m, self.n = (int(i) for i in input().split())

        self.c = np.zeros((self.n, 1))
        print("input c := ")
        self.c = np.array([eval(i) for i in input().split()], dtype=float).reshape((self.n, 1))

        self.A = np.zeros((self.m, self.n))
        print("input A := ")
        self.A = np.array([eval(i) for i in input().split()], dtype=float).reshape((self.m, self.n))

        self.b = np.zeros((self.m, 1))
        print("input b := ")
        self.b = np.array([eval(i) for i in input().split()], dtype=float).reshape((self.m, 1))
    
    def __print_status__(it, VNB, q, r, VB, p, theta, z):
        print("[ASP1]     Iteració{:>5} : q ={:>4}, rq ={:>10.3f}, B(p) ={:>4}, theta*={:>8.3f}, z ={:>10.3f}".format(it, VNB[q]+1, r[q,0], VB[p]+1, theta, z))
        
    def __simplex_body__(c, A, b, VB, VNB, xb, z, invB, fase, eps, it=0):
        it += 1
        
        # Identificació de SBF òptima i selecció de la VNB entrant B(q)
        r = (c[VNB].T - (c[VB].T@invB)@A[:,VNB]).T
        if (r >= -eps).all():
            if (fase == 'II'):
                print("[ASP1]     Iteració{:>5} : Solució òptima trobada.".format(it))
                te_solucio = True
            elif(z > eps):
                print("[ASP1]     Iteració{:>5} : Pe = Ø, (PL)e INFACTIBLE.".format(it))
                te_solucio = False
            else: 
                print("[ASP1]     Iteració{:>5} : Solució bàsica factible trobada.".format(it))
                te_solucio = True
            
            return te_solucio, VB, VNB, xb, z, invB, it, r
        
        aux = min([VNB[i] for i in np.where(r < -eps)[0]])
        q = np.where(VNB == aux)[0][0]
        
        # Càlcul de la DB de descens
        dB = (-invB@A[:,VNB[q]])
        dB.shape = (len(dB), 1)
        if (fase == 'II' and (dB >= -eps).all()):
            print("[ASP1]     Iteració{:>5} : (PL)e IL·LIMITAT.".format(it))
            te_solucio = False
            return te_solucio, VB, VNB, xb, z, invB, it, r
        
        # Càlcul de la passa màxima Ø* i selecció de la VB sortint B(p)
        theta, aux = min([(-xb[i,0]/dB[i,0], VB[i]) for i in np.where(dB < -eps)[0]])
        p = np.where(VB == aux)[0][0]
        
        # Actualitzacions i canvi de VB
        z += theta*r[q,0]
        
        Pe.__print_status__(it, VNB, q, r, VB, p, theta, z)

        xb += theta*dB
        xb[p] = theta
        VB[p], VNB[q] = VNB[q], VB[p]
        m = len(invB)
        aux = np.zeros((m,m))
        for i in range(m):
            if (i != p): aux[i] = invB[i] - dB[i]/dB[p]*invB[p]
            else: aux[i] = -1/dB[p]*invB[p]
        invB = aux
        
        # Tornar a l'inici
        return Pe.__simplex_body__(c, A, b, VB, VNB, xb, z, invB, fase, eps, it)

    def solve(self):
        print("[ASP1] Inici ASP1.")
        
        print("[ASP1]   Fase I")
        for i in range(self.m):
            if self.b[i] < 0:
                self.b[i] *= -1
                self.A[i] *= -1

        c = np.block([[np.zeros((self.n, 1))], [np.ones((self.m, 1))]])
        A = np.block([self.A, np.eye(self.m)])
        VB = np.arange(self.m) + self.n
        VNB = np.arange(self.n)
        invB = np.eye(self.m)
        xb = self.b.copy()
        z = np.sum(self.b)
        
        te_solucio, VB, VNB, xb, z, invB, it, _ = Pe.__simplex_body__(c, A, self.b, VB, VNB, xb, z, invB, 'I', self.eps)

        if te_solucio:
            print("[ASP1]   Fase II")
            VNB = VNB[VNB < self.n]
            z = (self.c[VB].T@xb)[0,0]
            te_solucio, VB, VNB, xb, z, _, _, r = Pe.__simplex_body__(self.c, self.A, self.b, VB, VNB, xb, z, invB, 'II', self.eps, it)        
            
        print("[ASP1] Fi ASP1", end='\n\n')
        
        if te_solucio:
            print("VB* = ")
            for i in range(len(VB)):
                print("{:>6}".format(VB[i]+1), end='')
            print(end='\n\n')
            
            print("xb* = ")
            for i in range(len(xb.T[0])):
                print("{:>10.4f}".format(xb.T[0,i]), end='')
            print(end='\n\n')
            
            print("VNB* = ")
            for i in range(len(VNB)):
                print("{:>6}".format(VNB[i]+1), end='')
            print(end='\n\n')
            
            print("r* = ")
            for i in range(len(r.T[0])):
                print("{:>10.4f}".format(r.T[0,i]), end='')
            print(end='\n\n')
            
            print("z* = ")
            print("{:>10.4f}".format(z))
        
        return